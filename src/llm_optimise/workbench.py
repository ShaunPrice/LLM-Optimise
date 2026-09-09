"""One feature API shared by the GUI and CLI; optional runtimes remain external."""

from __future__ import annotations

import dataclasses
import json
import uuid
from pathlib import Path

from .context import select_context
from .development import run_component, run_feedback
from .intelligence import Intelligence
from .quality import load_tasks
from .routing import RoutePolicy
from .runner import write_json
from .workspace import project_root, read_context

OPERATIONS = (
    "explore-plan",
    "explore-run",
    "dataset-prepare",
    "dataset-evaluate",
    "dataset-compare",
    "training-probe",
    "training-run",
    "adapter-register",
    "adapter-reload",
    "adapter-evaluate",
    "adapter-compare",
    "distill",
    "calibrate",
    "context",
    "feedback",
    "component",
    "specialist",
    "cache-clear",
    "status",
)
LONG_OPERATIONS = {
    "explore-plan",
    "explore-run",
    "training-probe",
    "training-run",
    "adapter-register",
    "adapter-reload",
    "adapter-evaluate",
    "distill",
    "calibrate",
    "feedback",
    "component",
    "specialist",
}


def workbench_status(workspace, engine=None):
    from .training_jobs import list_adapters

    root = Path(workspace).resolve()
    datasets = []
    paths = list((root / "runs" / "workbench").glob("*/manifest.json"))
    for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        if path.is_symlink() or path.stat().st_size > 16 * 1024**2:
            continue
        try:
            data = json.loads(path.read_text())
            if "splits" in data:
                datasets.append(
                    {
                        k: data.get(k)
                        for k in ("name", "dataset_sha256", "splits", "unique_rows", "warnings")
                    }
                    | {"manifest_path": str(path)}
                )
        except (ValueError, OSError):
            continue
    return {
        "operations": list(OPERATIONS),
        "datasets": datasets,
        "adapters": list_adapters(root / ".llm-optimise" / "adapters"),
        "intelligence": (engine or Intelligence(root)).status(),
    }


def execute_workbench(
    workspace,
    operation,
    data,
    *,
    models=(),
    engine=None,
    output_dir=None,
    cancel=None,
    progress=None,
    lease=None,
):
    if operation not in OPERATIONS or not isinstance(data, dict):
        raise ValueError("unknown workbench operation or invalid request")
    root = Path(workspace).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir
        else root / "runs" / "workbench" / uuid.uuid4().hex[:12]
    )
    engine = engine or Intelligence(root)
    registry = root / ".llm-optimise" / "adapters"

    def path(value):
        return str((root / str(value)).resolve())

    def rows(request):
        from .datasets import load_rows

        return load_rows(request["rows"] if "rows" in request else path(request["source"]))

    def generate(messages, max_tokens, max_cost_usd):
        policy = data.get("policy", {"placement": "local", "objective": "cost"})
        policy = dataclasses.asdict(policy) if isinstance(policy, RoutePolicy) else dict(policy)
        old = policy.get("max_cost_usd")
        policy["max_cost_usd"] = min(old, max_cost_usd) if old is not None else max_cost_usd
        return engine.run(
            list(models),
            messages,
            policy,
            data.get("selected_model"),
            max_tokens,
            "code" if operation == "feedback" else "chat",
            task_class=data.get("task_class", "general"),
            calibrated=data.get("calibrated", False),
            cancel=cancel,
            lease=lease,
        )

    if operation == "status":
        return workbench_status(root, engine)
    if operation == "cache-clear":
        return engine.clear_cache()
    if operation.startswith("explore-"):
        from .exploration import plan_exploration, run_exploration

        if operation == "explore-plan":
            return plan_exploration(data, base_dir=root)
        result = run_exploration(data, output, base_dir=root, cancel=cancel, progress=progress)
    elif operation == "dataset-prepare":
        from .datasets import prepare_dataset

        request = {**data}
        if "source" in request:
            request["source"] = path(request["source"])
        result = prepare_dataset(request, output)
    elif operation == "dataset-evaluate":
        from .datasets import evaluate_predictions

        result = evaluate_predictions(rows(data), data["predictions"])
    elif operation == "dataset-compare":
        from .datasets import compare_regression

        result = compare_regression(data["baseline"], data["candidate"], data.get("gates"))
    elif operation == "training-probe":
        from .training_jobs import probe_environment

        return probe_environment(path(data["python"]))
    elif operation == "training-run":
        from .training_jobs import run_training

        request = json.loads(json.dumps(data))
        request["python"] = path(request["python"])
        config = request["config"]
        config["base"] = path(config["base"])
        for key in ("train", "val"):
            if config.get("data", {}).get(key):
                config["data"][key] = path(config["data"][key])
        result = run_training(
            request, output, registry_dir=registry, cancel=cancel, progress=progress
        )
    elif operation == "adapter-register":
        from .training_jobs import register_adapter

        result = register_adapter(
            {
                **data,
                "path": path(data["path"]),
                "base_model": path(data["base_model"]),
                **({"python": path(data["python"])} if data.get("python") else {}),
            },
            registry,
        )
    elif operation in ("adapter-reload", "adapter-evaluate"):
        from .training_jobs import evaluate_adapter, reload_adapter

        request = {**data}
        if request.get("python"):
            request["python"] = path(request["python"])
        if operation == "adapter-evaluate":
            request["rows"] = rows(data)
        function = reload_adapter if operation == "adapter-reload" else evaluate_adapter
        result = function(request, registry, output, cancel=cancel, progress=progress)
    elif operation == "adapter-compare":
        from .training_jobs import compare_adapters

        result = compare_adapters(data["baseline"], data["candidate"], data.get("gates"))
    elif operation == "distill":
        from .distillation import distill

        result = distill(
            {**data, "rows": rows(data)}, output, generate, cancel=cancel, progress=progress
        )
    elif operation == "calibrate":
        tasks = load_tasks(path(data["dataset"]))
        result = engine.calibrate(
            list(models),
            tasks,
            task_class=data.get("task_class", "general"),
            model_ids=data.get("model_ids"),
            repeats=data.get("repeats", 1),
            max_requests=data.get("max_requests", 100),
            max_cost_usd=data.get("max_cost_usd", 0.1),
            max_tokens=data.get("max_tokens", 128),
            cancel=cancel,
            progress=progress,
            lease=lease,
        )
    elif operation == "context":
        files = read_context(project_root(root, data["project"]), data.get("context_files", []))
        result = select_context(
            data["query"],
            files,
            max_chars=data.get("max_chars", 12000),
            required_facts=data.get("required_facts", []),
        )
    elif operation == "feedback":
        result = run_feedback(
            project_root(root, data["project"]),
            output,
            data,
            generate,
            cancel=cancel,
            progress=progress,
        )
    elif operation == "component":
        result = run_component(project_root(root, data["project"]), output, data, cancel=cancel)
    elif operation == "specialist":
        result = engine.respond(
            list(models),
            [{"role": "user", "content": data["message"]}],
            contract=data["contract"],
            escalation_model=data.get("escalation_model"),
            policy=data.get("policy"),
            selected_model=data.get("selected_model"),
            max_tokens=data.get("max_tokens", 128),
            task_class=data.get("task_class", "general"),
            cache=data.get("cache", False),
            calibrated=data.get("calibrated", False),
            cancel=cancel,
            lease=lease,
        )
    write_json(output / "workbench-result.json", result)
    return {**result, "artifact_dir": str(output)}
