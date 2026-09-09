"""Command-line access to the same lab, router, agents and project tools as the GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import load_models
from .config import load_experiment
from .hardware import detect_hardware
from .intelligence import Intelligence
from .planner import estimate_memory
from .report import export_report
from .routing import RoutePolicy, choose_route
from .runner import run_experiment, write_json
from .training import save_recipe, training_recipe
from .workbench import OPERATIONS, execute_workbench
from .workspace import apply_proposal, code_messages, prepare_proposal, read_context


def emit(data):
    print(json.dumps(data, indent=2, allow_nan=False))


def parser():
    root = argparse.ArgumentParser(
        prog="llm-optimise",
        description="Measure and optimise specialised LLMs on constrained hardware.",
    )
    root.add_argument("--version", action="version", version="llm-optimise 0.1.0")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Detect hardware and installed runtimes")
    ui = sub.add_parser("ui", help="Launch the local graphical lab, development and chat interface")
    ui.add_argument("--workspace", default=".")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--open", action="store_true")
    ui.add_argument(
        "--host",
        choices=["127.0.0.1", "0.0.0.0"],
        default="127.0.0.1",
        help="0.0.0.0 is for Docker with a loopback-only published port",
    )
    run = sub.add_parser("run", help="Run a bounded experiment with live measurements")
    run.add_argument("config")
    run.add_argument("--output", required=True)
    run.add_argument("--resume", action="store_true")
    validate = sub.add_parser(
        "validate", help="Validate and expand an experiment without loading a model"
    )
    validate.add_argument("config")
    report = sub.add_parser("report", help="Export standalone HTML and CSV from measured results")
    report.add_argument("results")
    report.add_argument("--output", required=True)
    plan = sub.add_parser("estimate", help="Estimate dense-transformer weight and KV memory")
    for flag, default, kind in [
        ("parameters-b", 1.5, float),
        ("weight-bits", 4.5, float),
        ("layers", 28, int),
        ("kv-heads", 2, int),
        ("head-dim", 128, int),
        ("context", 2048, int),
        ("kv-bits", 16, float),
        ("concurrency", 1, int),
        ("overhead-gib", 0.5, float),
        ("gpu-fraction", 1, float),
    ]:
        plan.add_argument("--" + flag, type=kind, default=default)
    plan.add_argument("--unified", action="store_true")
    recipe = sub.add_parser("recipe", help="Prepare a Soup streaming, QLoRA or MLX training recipe")
    recipe.add_argument(
        "--engine", choices=["soup-stream", "soup-qlora", "soup-mlx"], required=True
    )
    recipe.add_argument("--model", required=True)
    recipe.add_argument("--data", required=True)
    recipe.add_argument("--model-output", default="./adapters")
    recipe.add_argument("--max-length", type=int, default=512)
    recipe.add_argument("--rank", type=int, default=8)
    recipe.add_argument("--output", required=True)
    for name in ("route", "agent", "code"):
        p = sub.add_parser(
            name,
            help={
                "route": "Explain a policy-based model choice without making an API request",
                "agent": "Send one request through the model router",
                "code": "Generate a reviewable code proposal using the chosen model",
            }[name],
        )
        p.add_argument(
            "--models",
            required=True,
            help="JSON model catalogue (keys referenced by environment variable name)",
        )
        p.add_argument("--placement", choices=["local", "cloud", "mixed"], default="local")
        p.add_argument("--objective", choices=["cost", "performance", "balanced"], default="cost")
        p.add_argument("--model", help="Pin model ID; still obey placement and hard constraints")
        p.add_argument("--max-cost-usd", type=float)
        p.add_argument("--max-latency-ms", type=float)
        p.add_argument("--min-quality", type=float)
        p.add_argument("--max-ram-gib", type=float)
        p.add_argument("--max-gpu-gib", type=float)
        p.add_argument("--max-tokens", type=int, default=1024)
        p.add_argument(
            "--workspace", default=".", help="Workspace for measurements and opt-in exact cache"
        )
        p.add_argument(
            "--task-class",
            default="general",
            help="Task family/version for scoped routing measurements",
        )
        p.add_argument(
            "--calibrated",
            action="store_true",
            help="Use matching measured latency and conservative quality confidence",
        )
        if name != "route":
            p.add_argument(
                "--cache",
                action="store_true",
                help="Opt into exact response caching at temperature zero",
            )
            p.add_argument(
                "--prefix-cache",
                action="store_true",
                help="Ask a compatible local endpoint to reuse its prompt prefix",
            )
        if name == "route":
            p.add_argument("--input-tokens", type=int, default=1000)
            p.add_argument("--capability", default="chat")
        else:
            p.add_argument("--message", required=True)
            p.add_argument("--output")
        if name == "code":
            p.add_argument("--project", required=True)
            p.add_argument("--context", nargs="*", default=[])
            p.add_argument(
                "--context-max-chars",
                type=int,
                help="Select relevant source chunks within this character budget",
            )
            p.add_argument(
                "--required-fact",
                action="append",
                default=[],
                help="Fact that must survive context selection; repeatable",
            )
    apply = sub.add_parser(
        "apply", help="Apply a previously reviewed proposal; refuses stale files"
    )
    apply.add_argument("proposal")
    apply.add_argument("--project", required=True)
    container = sub.add_parser(
        "container", help="Build/test an isolated project copy in resource-bounded Docker"
    )
    container.add_argument("--project", required=True)
    container.add_argument("--output", required=True)
    container.add_argument("--runtime", choices=["python", "node", "rust"], default="python")
    container.add_argument("--action", choices=["test", "build"], default="test")
    container.add_argument("--memory-mib", type=int, default=512)
    container.add_argument("--cpus", type=float, default=1)
    container.add_argument("--timeout-s", type=float, default=60)
    container.add_argument("--network", action="store_true")
    container.add_argument("--pull", action="store_true")
    container.add_argument("--image")
    workbench = sub.add_parser(
        "workbench",
        help="Run dataset, training, routing, component and repair workbench operations",
    )
    workbench.add_argument("operation", choices=OPERATIONS)
    workbench.add_argument(
        "--request", help="JSON request file; omitted only for status/cache-clear"
    )
    workbench.add_argument("--workspace", default=".")
    workbench.add_argument("--output", help="New output directory for reproducible evidence")
    workbench.add_argument("--models", help="Model registry; defaults to workspace settings")
    explore = sub.add_parser(
        "explore", help="Run capacity, KV, speculative, accelerator or multi-fidelity searches"
    )
    explore.add_argument("config")
    explore.add_argument("--workspace", default=".")
    explore.add_argument("--output")
    explore.add_argument(
        "--plan",
        action="store_true",
        help="Inspect capabilities and planned trials without model execution",
    )
    calibration = sub.add_parser(
        "calibrate", help="Measure chosen endpoints on a task dataset for empirical routing"
    )
    calibration.add_argument("--models", required=True)
    calibration.add_argument("--dataset", required=True)
    calibration.add_argument("--task-class", required=True)
    calibration.add_argument("--workspace", default=".")
    calibration.add_argument("--output")
    calibration.add_argument("--repeats", type=int, default=1)
    calibration.add_argument("--max-requests", type=int, default=100)
    calibration.add_argument("--max-cost-usd", type=float, default=0.1)
    calibration.add_argument("--max-tokens", type=int, default=128)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            emit(detect_hardware())
        elif args.command == "ui":
            from .server import serve

            serve(args.workspace, args.port, args.open, args.host)
        elif args.command in ("validate", "run"):
            experiment = load_experiment(args.config)
            from .quality import load_tasks

            tasks = load_tasks(experiment.dataset)
            if args.command == "validate":
                emit(
                    {
                        "valid": True,
                        "candidates": len(experiment.candidates),
                        "tasks": len(tasks),
                        "requests": len(experiment.candidates) * len(tasks) * experiment.repeats,
                        "note": "configuration validated; models and runtime have not been loaded",
                    }
                )
            else:

                def progress(event):
                    if event["phase"] != "measuring":
                        print(json.dumps(event, allow_nan=False), file=sys.stderr, flush=True)

                result = run_experiment(
                    experiment, args.output, resume=args.resume, progress=progress
                )
                path = export_report(result, args.output)
                emit(
                    {
                        "results": str(Path(args.output).resolve() / "results.json"),
                        "report": str(path),
                        "frontier": result["frontier"],
                        "complete": result["complete"],
                    }
                )
                return 0 if result["complete"] and result["frontier"] else 2
        elif args.command == "report":
            result = json.loads(Path(args.results).read_text(encoding="utf-8"))
            emit({"report": str(export_report(result, args.output))})
        elif args.command == "estimate":
            values = vars(args).copy()
            values.pop("command")
            emit(estimate_memory(**values))
        elif args.command == "recipe":
            recipe = training_recipe(
                args.engine,
                args.model,
                args.data,
                args.model_output,
                max_length=args.max_length,
                rank=args.rank,
            )
            emit(save_recipe(recipe, args.output))
        elif args.command in ("route", "agent", "code"):
            models = load_models(args.models)
            intelligence = Intelligence(args.workspace)
            policy = RoutePolicy(
                placement=args.placement,
                objective=args.objective,
                max_cost_usd=args.max_cost_usd,
                max_latency_ms=args.max_latency_ms,
                min_quality=args.min_quality,
                max_ram_gib=args.max_ram_gib,
                max_gpu_gib=args.max_gpu_gib,
            )
            if args.command == "route":
                if args.calibrated:
                    models, _ = intelligence.calibrated_models(
                        models, args.task_class, args.input_tokens
                    )
                result = choose_route(
                    models, policy, args.input_tokens, args.max_tokens, args.capability, args.model
                )
            else:
                if args.command == "code":
                    project = Path(args.project).resolve()
                    project.mkdir(parents=True, exist_ok=True)
                    context = read_context(project, args.context)
                    selection = None
                    if args.context_max_chars is not None:
                        from .context import select_context

                        selection = select_context(
                            args.message,
                            context,
                            max_chars=args.context_max_chars,
                            required_facts=args.required_fact,
                        )
                        if not selection["required_fact_gate"]:
                            raise ValueError("selected context omits required facts")
                    messages = code_messages(
                        args.message,
                        {"selected-context.txt": selection["text"]} if selection else context,
                    )
                else:
                    messages = [{"role": "user", "content": args.message}]
                result = intelligence.run(
                    models,
                    messages,
                    policy,
                    args.model,
                    args.max_tokens,
                    "code" if args.command == "code" else "chat",
                    task_class=args.task_class,
                    cache=args.cache,
                    calibrated=args.calibrated,
                    prefix_cache=args.prefix_cache,
                )
                if args.command == "code":
                    result["context_selection"] = selection
                    result["proposal"] = prepare_proposal(
                        project, result["completion"]["text"], context
                    )
                if args.output:
                    write_json(args.output, result)
            emit(result)
        elif args.command == "container":
            from .containers import run_container

            result = run_container(
                args.project,
                args.output,
                runtime=args.runtime,
                action=args.action,
                memory_mib=args.memory_mib,
                cpus=args.cpus,
                timeout_s=args.timeout_s,
                network=args.network,
                pull=args.pull,
                image=args.image,
            )
            write_json(Path(args.output) / "result.json", result)
            emit(result)
            return (
                0
                if result["exit_code"] == 0
                and not any(result[k] for k in ("timed_out", "cancelled", "output_limit_exceeded"))
                else 2
            )
        elif args.command == "apply":
            record = json.loads(Path(args.proposal).read_text(encoding="utf-8"))
            proposal = record.get("proposal", record)
            emit({"files": apply_proposal(Path(args.project).resolve(), proposal)})
            write_json(args.proposal, record)
        elif args.command in ("workbench", "explore", "calibrate"):

            def progress(event):
                print(json.dumps(event, allow_nan=False), file=sys.stderr, flush=True)

            models = []
            if args.command == "workbench":
                operation = args.operation
                if not args.request and operation not in {"status", "cache-clear"}:
                    raise ValueError("this workbench operation requires --request JSON")
                data = json.loads(Path(args.request).read_text()) if args.request else {}
                registry = (
                    Path(args.models)
                    if args.models
                    else Path(args.workspace) / ".llm-optimise" / "models.json"
                )
                models = load_models(registry) if registry.is_file() else []
            elif args.command == "explore":
                operation = "explore-plan" if args.plan else "explore-run"
                data = json.loads(Path(args.config).read_text())
            else:
                operation = "calibrate"
                data = {
                    key: getattr(args, key)
                    for key in (
                        "dataset",
                        "task_class",
                        "repeats",
                        "max_requests",
                        "max_cost_usd",
                        "max_tokens",
                    )
                }
                models = load_models(args.models)
            result = execute_workbench(
                args.workspace,
                operation,
                data,
                models=models,
                output_dir=args.output,
                progress=progress,
            )
            emit(result)
            if (
                result.get("passed") is False
                or result.get("accepted") is False
                or result.get("status") in {"failed", "cancelled", "timeout", "resource_limit"}
            ):
                return 2
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
