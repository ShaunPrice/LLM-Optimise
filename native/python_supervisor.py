"""Psutil reference implementation of the native JSONL process protocol.

This remains runnable without Rust. It is used for equal-workload profiling;
the core application's existing Backend is also fully Python and remains usable.
"""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

import psutil


def emit(value):
    print(json.dumps(value), flush=True)


def main():
    raw = sys.stdin.readline(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("configuration exceeds 1 MiB")
    config = json.loads(raw)
    command = config["command"]
    if not command or not all(isinstance(arg, str) and "\0" not in arg for arg in command):
        raise ValueError("command must contain strings")
    sample_ms, grace_ms = config.get("sample_ms", 50), config.get("grace_ms", 500)
    if not 5 <= sample_ms <= 10000 or not 0 <= grace_ms <= 10000:
        raise ValueError("invalid timing")
    events = queue.Queue()

    def read_control():
        for line in sys.stdin:
            if line.strip() == "cancel" or json.loads(line).get("command") == "cancel":
                events.put("cancelled")
                return
        if config.get("cancel_on_stdin_eof", True):
            events.put("controller_closed")

    threading.Thread(target=read_control, daemon=True).start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: events.put("signal"))
    log = open(config["log_path"], "wb") if config.get("log_path") else open(os.devnull, "wb")
    started = time.perf_counter()
    kwargs = (
        {"start_new_session": True}
        if os.name != "nt"
        else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    )
    worker = subprocess.Popen(
        command, cwd=config.get("cwd"), stdin=subprocess.DEVNULL, stdout=log, stderr=log, **kwargs
    )
    root = psutil.Process(worker.pid)
    peak = samples = 0
    known = {root.pid: root}
    reason = "error"
    emit(
        {
            "event": "started",
            "pid": worker.pid,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "supervisor": "python",
        }
    )
    try:
        while True:
            try:
                if root.status() == psutil.STATUS_ZOMBIE or not root.is_running():
                    reason = "exited"
                    break
            except psutil.NoSuchProcess:
                reason = "exited"
                break
            try:
                reason = events.get_nowait()
                break
            except queue.Empty:
                pass
            elapsed = (time.perf_counter() - started) * 1000
            if config.get("timeout_ms") and elapsed >= config["timeout_ms"]:
                reason = "timeout"
                break
            processes = [root, *root.children(recursive=True)]
            known.update({process.pid: process for process in processes})
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak, samples = max(peak, rss), samples + 1
            emit(
                {
                    "event": "sample",
                    "elapsed_ms": elapsed,
                    "rss_bytes": rss,
                    "processes": len(processes),
                }
            )
            if config.get("rss_limit_bytes") and rss > config["rss_limit_bytes"]:
                reason = "rss_limit"
                break
            time.sleep(sample_ms / 1000)
    finally:
        if os.name != "nt":
            for process in psutil.process_iter():
                try:
                    if os.getpgid(process.pid) == worker.pid:
                        known[process.pid] = process
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                for process in known.values():
                    try:
                        process.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            deadline = time.monotonic() + grace_ms / 1000
            while time.monotonic() < deadline:
                try:
                    if root.status() == psutil.STATUS_ZOMBIE:
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.01)
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                for process in known.values():
                    try:
                        process.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        else:
            for process in [*root.children(recursive=True), root]:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
        status = worker.wait(timeout=5)
        log.close()
    code = (
        status
        if reason == "exited" and status >= 0
        else (
            128 - status
            if reason == "exited"
            else 124
            if reason == "timeout"
            else 125
            if reason == "rss_limit"
            else 130
        )
    )
    emit(
        {
            "event": "finished",
            "reason": reason,
            "worker_exit_code": status if status >= 0 else None,
            "worker_signal": -status if status < 0 else None,
            "exit_code": code,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "rss_peak_bytes": peak,
            "samples": samples,
            "sample_ms": sample_ms,
            "memory_scope": "sampled owned process tree RSS; excludes GPU; short peaks may be missed",
        }
    )
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        emit({"event": "error", "error": str(exc)})
        sys.exit(2)
