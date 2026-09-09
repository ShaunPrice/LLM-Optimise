"""Actual matched-workload correctness and overhead profiling; no model/GPU claims."""

import json
import os
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "profile-local"
OUTPUT.mkdir(exist_ok=True)
RUST = ROOT / "target/release" / ("llm-supervisor.exe" if os.name == "nt" else "llm-supervisor")


def run(engine, scenario, repeat):
    worker = {
        "idle": "import time; time.sleep(0.8)",
        "rss_limit": "import time; x=bytearray(96*1024**2); time.sleep(10)",
        "cancel": "import subprocess,sys,time; c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(c.pid,flush=True); time.sleep(30)",
        "timeout": "import time; time.sleep(30)",
        "exit_code": "raise SystemExit(42)",
        "orphan": "import subprocess,sys; c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(c.pid,flush=True)",
    }[scenario]
    log = OUTPUT / f"{engine}-{scenario}-{repeat}.log"
    config = {
        "command": [sys.executable, "-u", "-c", worker],
        "sample_ms": 20,
        "grace_ms": 200,
        "log_path": str(log),
        "rss_limit_bytes": 48 * 1024**2 if scenario == "rss_limit" else 256 * 1024**2,
        "timeout_ms": 250 if scenario == "timeout" else 4000,
    }
    argv = (
        [str(RUST)]
        if engine == "rust"
        else [sys.executable, "-u", str(ROOT / "python_supervisor.py")]
    )
    start = time.perf_counter()
    child = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    measured = psutil.Process(child.pid)
    child.stdin.write(json.dumps(config) + "\n")
    child.stdin.flush()
    responses = queue.Queue()

    def reader():
        for line in child.stdout:
            responses.put(json.loads(line))

    threading.Thread(target=reader, daemon=True).start()
    peak = cpu = 0
    started_ms = cancel_start = cancelled_ms = None
    finished = None
    while time.perf_counter() - start < 8:
        try:
            peak = max(peak, measured.memory_info().rss)
            cpu = max(cpu, sum(measured.cpu_times()[:2]))
        except psutil.NoSuchProcess:
            pass
        try:
            response = responses.get(timeout=0.005)
        except queue.Empty:
            continue
        if response["event"] == "started":
            started_ms = (time.perf_counter() - start) * 1000
        if (
            scenario == "cancel"
            and started_ms is not None
            and cancel_start is None
            and time.perf_counter() - start > 0.25
        ):
            cancel_start = time.perf_counter()
            child.stdin.write('{"command":"cancel"}\n')
            child.stdin.flush()
        if response["event"] == "finished":
            finished = response
            if cancel_start:
                cancelled_ms = (time.perf_counter() - cancel_start) * 1000
            break
        if response["event"] == "error":
            child.wait(timeout=5)
            raise AssertionError((response, child.stderr.read()))
    if finished is None:
        child.kill()
        child.wait()
        raise AssertionError("supervisor did not finish")
    child.stdin.close()
    code = child.wait(timeout=3)
    expected = {
        "idle": ("exited", 0),
        "rss_limit": ("rss_limit", 125),
        "cancel": ("cancelled", 130),
        "timeout": ("timeout", 124),
        "exit_code": ("exited", 42),
        "orphan": ("exited", 0),
    }[scenario]
    assert (finished["reason"], code) == expected, (
        engine,
        scenario,
        finished,
        code,
        child.stderr.read(),
    )
    if scenario in ("cancel", "orphan"):
        descendant = int(log.read_text().strip())
        assert (
            not psutil.pid_exists(descendant)
            or psutil.Process(descendant).status() == psutil.STATUS_ZOMBIE
        )
    return {
        "engine": engine,
        "scenario": scenario,
        "repeat": repeat,
        "startup_ms": started_ms,
        "cancel_ms": cancelled_ms,
        "supervisor_peak_rss_bytes": peak,
        "supervisor_cpu_seconds": cpu,
        "worker_result": finished,
    }


records = []
for repeat in range(3):
    for scenario in ("idle", "rss_limit", "cancel", "timeout", "exit_code", "orphan"):
        for engine in ("rust", "python") if repeat % 2 == 0 else ("python", "rust"):
            records.append(run(engine, scenario, repeat))
summary = []
for scenario in ("idle", "rss_limit", "cancel", "timeout", "exit_code", "orphan"):
    for engine in ("rust", "python"):
        rows = [r for r in records if r["engine"] == engine and r["scenario"] == scenario]
        summary.append(
            {
                "engine": engine,
                "scenario": scenario,
                "startup_median_ms": statistics.median(r["startup_ms"] for r in rows),
                "cancel_median_ms": statistics.median(r["cancel_ms"] for r in rows)
                if scenario == "cancel"
                else None,
                "supervisor_peak_rss_median_mib": statistics.median(
                    r["supervisor_peak_rss_bytes"] for r in rows
                )
                / 1024**2,
                "supervisor_cpu_median_seconds": statistics.median(
                    r["supervisor_cpu_seconds"] for r in rows
                ),
            }
        )
result = {
    "platform": platform.platform(),
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "sample_ms": 20,
    "supervisor_rss_external_sample_ms": 5,
    "scope": "Supervisor-process overhead on identical small Python workers. Three sequential repeats in alternating implementation order; no LLM/GPU throughput claim.",
    "records": records,
    "summary": summary,
}
(OUTPUT / "results.json").write_text(json.dumps(result, indent=2))
print(json.dumps(summary, indent=2))
