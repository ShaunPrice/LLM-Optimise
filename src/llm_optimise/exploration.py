"""Bounded local capacity, successive-halving and inference workbenches.

Plans are JSON objects; each executed trial uses the existing runner and a fresh
managed process. Development tasks alone determine selection. An optional held-out
split is evaluated once after selection and never used to choose another model.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import shutil
import threading
import time
from pathlib import Path

from .backend import runtime_capabilities
from .config import Candidate, digest, file_digest, parse_experiment, positive
from .hardware import detect_hardware
from .quality import load_tasks
from .report import export_report
from .runner import pareto_frontier, run_experiment, write_json

MODES = ("capacity", "halving", "kv", "speculative", "accelerators")
DIMENSIONS = {"context": 131072, "batch_size": 8192, "ubatch_size": 8192, "threads": 256}


def _object(value, label, allowed):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"{label} must be an object with only {', '.join(sorted(allowed))}")
    return value


def _sequence(value, label, *, maximum, count=32):
    if not isinstance(value, list) or not value or len(value) > count:
        raise ValueError(f"{label} requires 1–{count} increasing values")
    for item in value:
        positive(item, label, integer=True)
        if item > maximum:
            raise ValueError(f"{label} exceeds {maximum}")
    if value != sorted(set(value)):
        raise ValueError(f"{label} values must be strictly increasing")
    return value


def kv_context_preset(experiment, contexts=None, cache_types=None):
    """Return ordinary experiment JSON; no inference or model download is performed."""
    result = json.loads(json.dumps(experiment, allow_nan=False))
    if len(result.get("candidates", [])) != 1 or "sweep" in result["candidates"][0]:
        raise ValueError("KV preset requires one unswept base candidate")
    contexts = _sequence(
        [512, 1024, 2048] if contexts is None else contexts, "contexts", maximum=131072
    )
    cache_types = ["f16", "q8_0", "q4_0"] if cache_types is None else cache_types
    if (
        not isinstance(cache_types, list)
        or not cache_types
        or len(set(cache_types)) != len(cache_types)
    ):
        raise ValueError("cache_types must be a nonempty unique list")
    if any(kind not in ("f16", "q8_0", "q4_0") for kind in cache_types):
        raise ValueError("unsupported KV cache type")
    base = result["candidates"][0]
    result["candidates"] = [
        {
            **base,
            "name": f"kv-{context}-{kind}",
            "context": context,
            "cache_type_k": kind,
            "cache_type_v": kind,
            "flash_attention": "on" if kind != "f16" else base.get("flash_attention", "off"),
        }
        for context in contexts
        for kind in cache_types
    ]
    if len(result["candidates"]) > result.get("max_trials", 32):
        raise ValueError("KV preset exceeds max_trials")
    return result


def _base(spec, base_dir):
    experiment = parse_experiment(spec["experiment"], base_dir)
    if any(c.backend != "llama.cpp" for c in experiment.candidates):
        raise ValueError(
            "exploration requires managed local llama.cpp candidates; no external endpoints"
        )
    if experiment.max_trials > 64 or experiment.repeats > 10 or experiment.max_tokens > 2048:
        raise ValueError("exploration bounds: max_trials<=64, repeats<=10, max_tokens<=2048")
    if max(experiment.timeout_s, experiment.startup_timeout_s) > 300:
        raise ValueError("exploration request and startup timeouts must be <=300 seconds")
    if experiment.limits.min_available_gib <= 0:
        raise ValueError("exploration requires a positive min_available_gib headroom reserve")
    return experiment


def _replace_candidates(experiment, candidates):
    return dataclasses.replace(experiment, candidates=tuple(candidates))


def _split_identity(task):
    return digest({"prompt": task.prompt.strip(), "system": task.system.strip()})


def plan_exploration(spec, *, base_dir="."):
    """Validate and expand a JSON workbench spec, including installed-runtime evidence.

    Modes: capacity{dimensions,recovery_timeout_s,verify_recovery};
    search{task_budgets,eta,heldout_dataset} for halving; kv{contexts,cache_types};
    speculative{draft_models,draft_max}; accelerators[{accelerator,executable,device?}].
    No endpoint requests, model loading, training or held-out scoring occur here.
    """
    _object(
        spec,
        "exploration",
        {
            "mode",
            "experiment",
            "capacity",
            "search",
            "kv",
            "speculative",
            "accelerators",
            "max_duration_s",
        },
    )
    mode = spec.get("mode")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    duration = spec.get("max_duration_s", 900)
    positive(duration, "max_duration_s")
    if duration > 3600:
        raise ValueError("max_duration_s must be <=3600")
    experiment = _base(spec, base_dir)
    development = load_tasks(experiment.dataset)
    candidates = list(experiment.candidates)
    excluded, capabilities, options = [], {}, {}
    if mode != "halving" and len(candidates) != 1:
        raise ValueError(f"{mode} requires one base candidate")
    base = candidates[0]
    if mode == "capacity":
        settings = _object(
            spec.get("capacity", {}),
            "capacity",
            {"dimensions", "recovery_timeout_s", "verify_recovery"},
        )
        dimensions = _object(
            settings.get("dimensions", {"context": [512, 1024, 2048, 4096]}),
            "dimensions",
            DIMENSIONS,
        )
        if not dimensions:
            raise ValueError("capacity requires at least one dimension")
        for name, values in dimensions.items():
            _sequence(values, name, maximum=DIMENSIONS[name])
        steps = max(map(len, dimensions.values()))
        candidates = []
        for step in range(steps):
            values = {
                name: options[min(step, len(options) - 1)] for name, options in dimensions.items()
            }
            if "batch_size" in values and "ubatch_size" not in values:
                values["ubatch_size"] = min(base.ubatch_size, values["batch_size"])
            candidates.append(dataclasses.replace(base, name=f"capacity-{step + 1:03d}", **values))
        recovery = settings.get("recovery_timeout_s", 10)
        positive(recovery, "recovery_timeout_s")
        if recovery > 60 or type(settings.get("verify_recovery", True)) is not bool:
            raise ValueError("recovery_timeout_s must be <=60 and verify_recovery boolean")
        options = {
            "dimensions": dimensions,
            "recovery_timeout_s": recovery,
            "verify_recovery": settings.get("verify_recovery", True),
        }
    elif mode == "kv":
        settings = _object(spec.get("kv", {}), "kv", {"contexts", "cache_types"})
        prepared = kv_context_preset(
            dataclasses.asdict(experiment), settings.get("contexts"), settings.get("cache_types")
        )
        candidates = list(parse_experiment(prepared, base_dir).candidates)
    elif mode == "speculative":
        settings = _object(
            spec.get("speculative", {}), "speculative", {"draft_models", "draft_max"}
        )
        drafts = settings.get("draft_models", [])
        if (
            not isinstance(drafts, list)
            or not drafts
            or len(drafts) > 8
            or any(not isinstance(p, str) or not p for p in drafts)
        ):
            raise ValueError("speculative requires 1–8 local draft model paths")
        lengths = _sequence(settings.get("draft_max", [4, 8]), "draft_max", maximum=64)
        cap = runtime_capabilities(base.executable)
        capabilities[base.executable] = cap
        flags = cap["flags"]
        if (
            not cap.get("help_available")
            or not any(f in flags for f in ("--spec-draft-model", "--model-draft"))
            or not any(f in flags for f in ("--spec-draft-n-max", "--draft-max"))
        ):
            raise ValueError(
                "installed runtime does not advertise draft-model and draft-length support"
            )
        if base.gpu_layers == 0 and (
            not any(f in flags for f in ("--spec-draft-ngl", "--gpu-layers-draft"))
            or not any(f in flags for f in ("--spec-draft-device", "--device-draft"))
        ):
            raise ValueError("runtime cannot explicitly keep the draft model on CPU")
        candidates = [
            dataclasses.replace(base, name="spec-baseline", draft_model=None, draft_max=None)
        ]
        for index, draft in enumerate(drafts):
            path = (Path(base_dir) / draft).resolve()
            if not path.is_file():
                raise ValueError(f"draft model file not found: {path}")
            candidates.extend(
                dataclasses.replace(
                    base,
                    name=f"spec-draft-{index + 1}-{length}",
                    draft_model=str(path),
                    draft_max=length,
                )
                for length in lengths
            )
        options = {
            "acceptance_note": "Explicit terminal draft_n/draft_n_accepted counters only; unavailable when the runtime does not expose them."
        }
    elif mode == "accelerators":
        entries = spec.get("accelerators", [])
        if not isinstance(entries, list) or not entries or len(entries) > 16:
            raise ValueError("accelerators requires 1–16 explicit runtime entries")
        candidates = []
        for index, entry in enumerate(entries):
            _object(entry, "accelerator", {"accelerator", "executable", "device"})
            accelerator = entry.get("accelerator")
            if accelerator not in ("cpu", "cuda", "metal", "vulkan", "rocm"):
                raise ValueError("choose cpu, cuda, metal, vulkan or rocm")
            executable = entry.get("executable")
            if not isinstance(executable, str) or not any(c in executable for c in ("/", "\\")):
                raise ValueError(
                    "accelerator comparisons require explicit installed executable paths"
                )
            executable = str((Path(base_dir) / executable).resolve())
            cap = runtime_capabilities(executable)
            capabilities[executable] = cap
            if accelerator not in cap["supported_accelerators"]:
                excluded.append(
                    {
                        "accelerator": accelerator,
                        "executable": executable,
                        "reason": "runtime/device support unavailable or unverified",
                    }
                )
                continue
            if accelerator != "cpu" and "--device" not in cap["flags"]:
                excluded.append(
                    {
                        "accelerator": accelerator,
                        "executable": executable,
                        "reason": "runtime cannot explicitly select the requested accelerator device",
                    }
                )
                continue
            devices = [d["id"] for d in cap["devices"] if d["accelerator"] == accelerator]
            device = entry.get("device")
            if device is not None and device not in devices:
                raise ValueError("device not enumerated for the requested accelerator")
            candidates.append(
                dataclasses.replace(
                    base,
                    name=f"accelerator-{index + 1}-{accelerator}",
                    executable=executable,
                    accelerator=accelerator,
                    gpu_layers=0 if accelerator == "cpu" else -1,
                    device=None if accelerator == "cpu" else (device or devices[0]),
                )
            )
    else:
        settings = _object(
            spec.get("search", {}), "search", {"task_budgets", "eta", "heldout_dataset"}
        )
        budgets = settings.get("task_budgets", sorted({min(2, len(development)), len(development)}))
        budgets = _sequence(budgets, "task_budgets", maximum=len(development), count=8)
        if budgets[-1] != len(development):
            budgets = [*budgets, len(development)]
        eta = settings.get("eta", 2)
        positive(eta, "eta", integer=True)
        if not 2 <= eta <= 8:
            raise ValueError("eta must be 2–8")
        heldout = settings.get("heldout_dataset")
        if heldout is not None:
            heldout = str((Path(base_dir) / heldout).resolve())
            tasks = load_tasks(heldout)
            if ({t.id for t in tasks} & {t.id for t in development}) or (
                {_split_identity(t) for t in tasks} & {_split_identity(t) for t in development}
            ):
                raise ValueError("held-out split overlaps development task IDs or prompts")
        options = {
            "task_budgets": budgets,
            "eta": eta,
            "heldout_dataset": heldout,
            "heldout_sha256": file_digest(heldout) if heldout else None,
        }
    if candidates:
        experiment = _replace_candidates(experiment, candidates)
    evaluations = len(candidates)
    if mode == "halving":
        survivors, evaluations = len(candidates), 0
        requests = 0
        for budget in options["task_budgets"]:
            evaluations += survivors
            requests += survivors * (budget * experiment.repeats + experiment.warmup)
            survivors = max(1, math.ceil(survivors / options["eta"]))
        if options["heldout_dataset"]:
            evaluations += 1
            requests += (
                len(load_tasks(options["heldout_dataset"])) * experiment.repeats + experiment.warmup
            )
    else:
        if mode == "capacity" and options["verify_recovery"]:
            evaluations += 1
        requests = evaluations * (len(development) * experiment.repeats + experiment.warmup)
    if evaluations > 128 or requests > 10000:
        raise ValueError("exploration exceeds 128 worker evaluations or 10000 requests")
    input_paths = {experiment.dataset}
    for candidate in candidates:
        input_paths.add(candidate.model)
        if candidate.draft_model:
            input_paths.add(candidate.draft_model)
        executable = shutil.which(candidate.executable)
        if executable:
            input_paths.add(executable)
    hashes = {str(Path(path).resolve()): file_digest(path) for path in sorted(input_paths)}
    return {
        "schema_version": 1,
        "mode": mode,
        "experiment": dataclasses.asdict(experiment),
        "candidates": [dataclasses.asdict(c) for c in candidates],
        "excluded_candidates": excluded,
        "runtime_capabilities": capabilities,
        "options": options,
        "max_duration_s": duration,
        "max_worker_evaluations": evaluations,
        "max_requests_including_warmup": requests,
        "development_sha256": file_digest(experiment.dataset),
        "input_sha256": hashes,
        "memory_enforcement": "sampled cooperative guard; not an OS hard memory cap",
        "selection_data": "development only; held-out results never feed selection",
    }


def classify_trial(trial, log_text=""):
    """Classify observed failure evidence without inferring an OOM from any crash."""
    if trial.get("status") == "cancelled":
        return "cancelled"
    reasons = " ".join(trial.get("rejection_reasons", []))
    message = (str(trial.get("error", "")) + " " + reasons + " " + log_text).lower()
    if trial.get("memory", {}).get("violation") or "insufficient available ram" in message:
        return "memory_guard"
    if any(s in message for s in ("out of memory", "failed to allocate", "cannot allocate memory")):
        return "allocation_failure"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "unavailable" in reasons.lower():
        return "telemetry_unavailable"
    if "quality below" in message:
        return "quality_gate"
    if "latency" in message or "time to first token" in message:
        return "latency_gate"
    if any(
        s in message for s in ("unsupported", "does not advertise", "cannot verify accelerator")
    ):
        return "unsupported_configuration"
    if trial.get("status") != "complete":
        return "runtime_failure"
    return "eligible" if trial.get("eligible") else "constraint_gate"


def _rank(trials):
    # Quality first; unobserved memory never receives a favourable zero estimate.
    return sorted(
        (t for t in trials if t.get("eligible")),
        key=lambda t: (
            -t["metrics"]["quality"],
            t["metrics"]["latency_p50_s"],
            t["memory"].get("rss_peak_gib")
            if t["memory"].get("rss_peak_gib") is not None
            else math.inf,
            t["name"],
        ),
    )


def _wait_headroom(reserve, timeout, cancel):
    deadline = time.monotonic() + timeout
    while not cancel.is_set():
        available = detect_hardware()["ram_available_gib"]
        if available >= reserve:
            return {"recovered": True, "available_gib": available, "required_gib": reserve}
        if time.monotonic() >= deadline:
            return {"recovered": False, "available_gib": available, "required_gib": reserve}
        cancel.wait(min(0.1, max(0, deadline - time.monotonic())))
    return {"recovered": False, "cancelled": True, "required_gib": reserve}


def run_exploration(spec, output_dir, *, base_dir=".", cancel=None, progress=None):
    """Execute a validated workbench and persist exploration.json plus ordinary runs.

    Returns a JSON record with plan, stages, trials, frontier, selected_candidate,
    heldout, boundary, recovery, complete, cancelled and deadline_exceeded.
    A deadline terminates the owned managed worker through runner cancellation.
    """
    cancel = cancel or threading.Event()
    progress = progress or (lambda event: None)
    plan = plan_exploration(spec, base_dir=base_dir)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "exploration output directory must be new or empty; resume is not supported"
        )
    output.mkdir(parents=True, exist_ok=True)
    experiment = parse_experiment(plan["experiment"])
    candidates = [Candidate(**c) for c in plan["candidates"]]
    result = {
        "schema_version": 1,
        "mode": plan["mode"],
        "plan": plan,
        "stages": [],
        "trials": [],
        "frontier": [],
        "selected_candidate": None,
        "heldout": None,
        "boundary": None,
        "recovery": None,
        "complete": False,
        "cancelled": False,
        "deadline_exceeded": False,
    }
    write_json(output / "plan.json", plan)
    write_json(output / "exploration.json", result)
    started = time.monotonic()
    finished = threading.Event()

    def deadline_guard():
        if not finished.wait(plan["max_duration_s"]):
            result["deadline_exceeded"] = True
            cancel.set()

    watchdog = threading.Thread(target=deadline_guard, daemon=True)
    watchdog.start()

    def execute(candidate_list, label, dataset=None):
        if cancel.is_set():
            return []
        # Recheck original data before every stage; the plan cannot silently drift.
        for source, expected in plan["input_sha256"].items():
            if file_digest(source) != expected:
                raise ValueError(
                    "model, runtime or development dataset changed after exploration planning"
                )
        stage_exp = dataclasses.replace(
            experiment,
            name=f"{experiment.name}-{label}",
            dataset=dataset or experiment.dataset,
            candidates=tuple(candidate_list),
        )
        directory = output / label
        progress({"phase": "exploration_stage", "stage": label, "candidates": len(candidate_list)})
        observed = run_experiment(
            stage_exp,
            directory,
            cancel=cancel,
            progress=lambda event: progress({**event, "stage": label}),
        )
        export_report(observed, directory)
        for trial in observed["trials"]:
            logs = ""
            if trial["status"] != "complete":
                # Only the run's own bounded tail contributes to failure diagnosis.
                log_name = observed.get("execution_order", []).index(trial["name"])
                log = directory / f"trial-{log_name:03d}.log"
                if log.exists():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 65536))
                        logs = stream.read(65536).decode("utf-8", errors="replace")
            trial["failure_class"] = classify_trial(trial, logs)
        result["stages"].append(
            {
                "name": label,
                "dataset_sha256": file_digest(stage_exp.dataset),
                "task_count": len(load_tasks(stage_exp.dataset)),
                "results_path": str(directory / "results.json"),
                "trials": observed["trials"],
                "complete": observed["complete"],
            }
        )
        write_json(output / "exploration.json", result)
        return observed["trials"]

    try:
        if not candidates:
            result["boundary"] = {
                "reason": "no_verified_runtime",
                "excluded_candidates": plan["excluded_candidates"],
            }
        elif plan["mode"] == "halving":
            tasks = load_tasks(experiment.dataset)
            random.Random(experiment.seed).shuffle(tasks)
            survivors = candidates
            for stage, budget in enumerate(plan["options"]["task_budgets"]):
                if cancel.is_set() or not survivors:
                    break
                dataset = output / f"development-{stage:02d}.jsonl"
                dataset.write_text(
                    "".join(
                        json.dumps(dataclasses.asdict(t), allow_nan=False) + "\n"
                        for t in tasks[:budget]
                    ),
                    encoding="utf-8",
                )
                trials = execute(survivors, f"stage-{stage:02d}", str(dataset))
                result["trials"] = trials
                ranked = _rank(trials)
                keep = max(1, math.ceil(len(survivors) / plan["options"]["eta"]))
                ids = {t["name"] for t in ranked[:keep]}
                survivors = [c for c in survivors if c.name in ids]
                result["stages"][-1]["promoted"] = [c.name for c in survivors]
            ranked = _rank(result["trials"])
            if ranked and not cancel.is_set():
                result["selected_candidate"] = ranked[0]["name"]
                heldout = plan["options"]["heldout_dataset"]
                if heldout:
                    if file_digest(heldout) != plan["options"]["heldout_sha256"]:
                        raise ValueError("held-out dataset changed after planning")
                    winner = next(c for c in candidates if c.name == result["selected_candidate"])
                    rows = execute([winner], "heldout-final", heldout)
                    result["heldout"] = {
                        "candidate": winner.name,
                        "trials": rows,
                        "used_for_selection": False,
                        "passed": bool(rows and rows[0].get("eligible")),
                    }
        elif plan["mode"] == "capacity":
            last_good = None
            for index, candidate in enumerate(candidates):
                if cancel.is_set():
                    break
                rows = execute([candidate], f"capacity-{index:03d}")
                result["trials"].extend(rows)
                if not rows:
                    break
                trial = rows[0]
                if trial.get("eligible"):
                    last_good = candidate
                    result["selected_candidate"] = candidate.name
                else:
                    result["boundary"] = {
                        "candidate": candidate.name,
                        "failure_class": trial["failure_class"],
                        "rejection_reasons": trial.get("rejection_reasons", []),
                        "interpretation": "first observed boundary along this configured staircase; not a global capacity limit",
                    }
                    result["recovery"] = _wait_headroom(
                        experiment.limits.min_available_gib,
                        plan["options"]["recovery_timeout_s"],
                        cancel,
                    )
                    if (
                        last_good
                        and result["recovery"]["recovered"]
                        and plan["options"]["verify_recovery"]
                        and not cancel.is_set()
                    ):
                        recovered = execute([last_good], "recovery-control")
                        result["recovery"]["control_passed"] = bool(
                            recovered and recovered[0].get("eligible")
                        )
                        result["recovery"]["control_candidate"] = last_good.name
                    break
        else:
            result["trials"] = execute(candidates, "comparison")
            ranked = _rank(result["trials"])
            if ranked:
                result["selected_candidate"] = ranked[0]["name"]
            if plan["mode"] == "speculative":
                baseline = next(
                    (
                        t
                        for t in result["trials"]
                        if t["name"] == "spec-baseline" and t.get("eligible")
                    ),
                    None,
                )
                for trial in result["trials"]:
                    latency = trial.get("metrics", {}).get("latency_p50_s")
                    trial["speedup_vs_baseline"] = (
                        baseline["metrics"]["latency_p50_s"] / latency
                        if baseline and trial.get("eligible") and latency
                        else None
                    )
        result["frontier"] = pareto_frontier(result["trials"])
        result["complete"] = not cancel.is_set() and bool(candidates)
        result["selection_passed"] = result["selected_candidate"] is not None
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finished.set()
        watchdog.join(timeout=1)
        result["cancelled"] = cancel.is_set()
        result["complete"] = result["complete"] and not result["cancelled"]
        result["elapsed_s"] = time.monotonic() - started
        write_json(output / "exploration.json", result)
    progress(
        {
            "phase": "exploration_finished",
            "complete": result["complete"],
            "selected_candidate": result["selected_candidate"],
            "frontier": result["frontier"],
        }
    )
    return result
