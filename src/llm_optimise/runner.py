"""Bounded, resumable experiments with quality gates and explicit evidence provenance."""

from __future__ import annotations

import dataclasses
import json
import math
import random
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .backend import Backend
from .config import digest, file_digest
from .hardware import ResourceMonitor, detect_hardware
from .quality import load_tasks


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def quantile(values, fraction):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def aggregate(samples):
    latencies = [s["latency_s"] for s in samples]
    tokens = [s["output_tokens"] for s in samples]
    complete_counts = all(n is not None for n in tokens)
    decode_seconds = [
        s["output_tokens"] / s["decode_tokens_s"]
        for s in samples
        if s["output_tokens"] and s["decode_tokens_s"] and s["decode_tokens_s"] > 0
    ]
    decode = (
        sum(tokens) / sum(decode_seconds)
        if complete_counts and len(decode_seconds) == len(samples) and decode_seconds
        else None
    )
    speculation_known = all(
        type(s.get("draft_tokens")) is int and type(s.get("accepted_draft_tokens")) is int
        for s in samples
    )
    drafted = sum(s["draft_tokens"] for s in samples) if speculation_known else None
    accepted = sum(s["accepted_draft_tokens"] for s in samples) if speculation_known else None
    return {
        "quality": statistics.mean(s["score"] for s in samples),
        "latency_p50_s": quantile(latencies, 0.5),
        "latency_p95_s": quantile(latencies, 0.95),
        "ttft_p50_s": quantile([s["ttft_s"] for s in samples], 0.5),
        "ttft_p95_s": quantile([s["ttft_s"] for s in samples], 0.95),
        "decode_tokens_s": decode,
        "end_to_end_tokens_s": sum(tokens) / sum(latencies)
        if complete_counts and sum(latencies) > 0
        else None,
        "requests": len(samples),
        "truncated_requests": sum(bool(s["truncated"]) for s in samples),
        "draft_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "draft_acceptance_rate": accepted / drafted if drafted else None,
        "quality_by_task": {
            key: statistics.mean(s["score"] for s in samples if s["task_id"] == key)
            for key in sorted({s["task_id"] for s in samples})
        },
    }


def eligibility(trial, limits):
    if trial["status"] != "complete":
        return [trial.get("error", trial["status"])]
    metrics, memory = trial["metrics"], trial["memory"]
    reasons = []
    if metrics["quality"] < limits.min_quality:
        reasons.append("quality below minimum")
    checks = [
        (limits.max_rss_gib, memory["rss_peak_gib"], "RSS"),
        (limits.max_gpu_gib, memory["gpu_peak_gib"], "GPU memory"),
        (limits.max_latency_s, metrics["latency_p95_s"], "p95 latency"),
        (limits.max_ttft_s, metrics["ttft_p95_s"], "p95 time to first token"),
    ]
    for bound, observed, label in checks:
        if bound is not None and (observed is None or observed > bound):
            reasons.append(
                f"{label} unavailable" if observed is None else f"{label} exceeded budget"
            )
    if memory.get("violation"):
        reasons.append(memory["violation"])
    return reasons


def pareto_frontier(trials):
    eligible = [t for t in trials if t.get("eligible")]

    def vector(t):
        return (
            -t["metrics"]["quality"],
            t["metrics"]["latency_p50_s"],
            t["memory"]["rss_peak_gib"],
            t["memory"]["gpu_peak_gib"],
        )

    def dominates(a, b):
        # Never infer superiority from absent telemetry. Compare identical observable dimensions.
        if any((x is None) != (y is None) for x, y in zip(a, b, strict=False)):
            return False
        pairs = [(x, y) for x, y in zip(a, b, strict=False) if x is not None]
        return all(x <= y for x, y in pairs) and any(x < y for x, y in pairs)

    return [
        t["name"]
        for t in eligible
        if not any(dominates(vector(other), vector(t)) for other in eligible if other is not t)
    ]


def _provenance(experiment):
    import shutil

    files = {experiment.dataset}
    for c in experiment.candidates:
        if c.backend == "llama.cpp":
            files.add(c.model)
            if c.draft_model:
                files.add(c.draft_model)
            exe = shutil.which(c.executable)
            if exe:
                files.add(exe)
    return {str(path): file_digest(path) for path in sorted(files)}


def run_experiment(experiment, output_dir, *, resume=False, cancel=None, progress=None):
    cancel = cancel or threading.Event()
    progress = progress or (lambda event: None)
    tasks = load_tasks(experiment.dataset)
    output = Path(output_dir).resolve()
    manifest_path = output / "results.json"
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("output directory is not empty; use --resume or a new directory")
    progress({"phase": "preparing", "message": "Hashing models and dataset for reproducibility"})
    hashes = _provenance(experiment)
    spec = dataclasses.asdict(experiment)
    fingerprint = digest({"experiment": spec, "files": hashes})
    if resume and any(c.backend == "openai" for c in experiment.candidates):
        raise ValueError(
            "external endpoint identity can change; resume requires managed llama.cpp models"
        )
    if resume and manifest_path.exists():
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError(
                "experiment, runtime or input files changed; choose a new output directory"
            )
    else:
        result = {
            "schema_version": 1,
            "name": experiment.name,
            "fingerprint": fingerprint,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "experiment": spec,
            "file_sha256": hashes,
            "hardware": detect_hardware(),
            "trials": [],
            "frontier": [],
        }
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, result)
    complete = {t["name"] for t in result["trials"] if t["status"] == "complete"}
    candidates = list(experiment.candidates)
    random.Random(experiment.seed).shuffle(candidates)
    result["execution_order"] = [c.name for c in candidates]
    try:
        for index, candidate in enumerate(candidates):
            if cancel.is_set():
                break
            if candidate.name in complete:
                continue
            result["trials"] = [t for t in result["trials"] if t["name"] != candidate.name]
            progress(
                {
                    "phase": "running",
                    "candidate": candidate.name,
                    "completed": len(complete),
                    "total": len(candidates),
                }
            )
            trial = {
                "name": candidate.name,
                "candidate": dataclasses.asdict(candidate),
                "status": "running",
                "metrics": {},
                "memory": {},
            }
            backend, monitor = None, None
            guard_stop, guard = threading.Event(), None
            samples = []
            started = time.monotonic()
            try:
                if detect_hardware()["ram_available_gib"] < experiment.limits.min_available_gib:
                    raise RuntimeError("insufficient available RAM before launch")
                backend = Backend(candidate, output / f"trial-{index:03d}.log")
                backend.launch()
                monitor = ResourceMonitor(
                    backend.pid,
                    experiment.limits,
                    cpu_only=candidate.backend == "llama.cpp" and candidate.gpu_layers == 0,
                ).start()

                def guard_worker(backend=backend, monitor=monitor, stop=guard_stop):
                    # A managed worker is terminated even if its HTTP stream is stalled.
                    # This is still sampled resource enforcement, not an OS hard memory cap.
                    while not stop.wait(0.05):
                        if cancel.is_set() or monitor.violation:
                            if backend.pid is not None:
                                backend.close()
                            return

                guard = threading.Thread(target=guard_worker, daemon=True)
                guard.start()

                def check(monitor=monitor):
                    if cancel.is_set():
                        raise InterruptedError("cancelled by user")
                    if monitor.violation:
                        raise MemoryError(monitor.violation)

                backend.ready(experiment.startup_timeout_s, check)
                trial["load_s"] = time.monotonic() - started
                for _ in range(experiment.warmup):
                    check()
                    backend.generate(
                        tasks[0],
                        experiment.max_tokens,
                        experiment.seed,
                        experiment.timeout_s,
                        check,
                    )
                with open(output / f"trial-{index:03d}.jsonl", "w", encoding="utf-8") as stream:
                    for repeat in range(experiment.repeats):
                        order = list(tasks)
                        random.Random(experiment.seed + repeat).shuffle(order)
                        for task in order:
                            check()
                            generation = backend.generate(
                                task,
                                experiment.max_tokens,
                                experiment.seed,
                                experiment.timeout_s,
                                check,
                            )
                            check()
                            sample = {
                                **dataclasses.asdict(generation),
                                "task_id": task.id,
                                "repeat": repeat,
                                "score": task.score(generation.text),
                            }
                            samples.append(sample)
                            stream.write(json.dumps(sample, allow_nan=False) + "\n")
                            stream.flush()
                            progress(
                                {
                                    "phase": "measuring",
                                    "candidate": candidate.name,
                                    "requests": len(samples),
                                    "request_total": len(tasks) * experiment.repeats,
                                }
                            )
                trial["metrics"] = aggregate(samples)
                trial["status"] = "complete"
                complete.add(candidate.name)
            except (Exception, KeyboardInterrupt) as exc:
                if isinstance(exc, KeyboardInterrupt):
                    cancel.set()
                trial["status"] = "cancelled" if cancel.is_set() else "failed"
                # Error messages come from runtime/HTTP, never persist credentials or request headers.
                error = f"{type(exc).__name__}: {exc}"
                if backend and backend.key:
                    error = error.replace(backend.key, "<redacted>")
                trial["error"] = error
                trial["partial_requests"] = len(samples)
            finally:
                guard_stop.set()
                if guard:
                    guard.join(timeout=10)
                if monitor:
                    trial["memory"] = monitor.stop()
                if backend:
                    trial["runtime_version"] = backend.version
                    trial["command"] = backend.command
                    backend.close()
                trial["elapsed_s"] = time.monotonic() - started
            trial["rejection_reasons"] = eligibility(trial, experiment.limits)
            trial["eligible"] = not trial["rejection_reasons"]
            result["trials"].append(trial)
            result["frontier"] = pareto_frontier(result["trials"])
            write_json(manifest_path, result)
            progress({"phase": "trial_complete", "trial": trial})
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        result["cancelled"] = cancel.is_set()
        result["complete"] = len(result["trials"]) == len(candidates) and all(
            t["status"] == "complete" for t in result["trials"]
        )
        write_json(manifest_path, result)
    progress({"phase": "finished", "frontier": result["frontier"], "complete": result["complete"]})
    return result
