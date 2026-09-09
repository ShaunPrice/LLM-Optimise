"""Real Windows job-object protocol validation using only Python's standard library."""

import ctypes
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / "profile-local"
out.mkdir(exist_ok=True)
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.restype = ctypes.c_void_p
kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
kernel.CloseHandle.argtypes = [ctypes.c_void_p]


def alive(pid):
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        status = ctypes.c_ulong()
        assert kernel.GetExitCodeProcess(handle, ctypes.byref(status))
        return status.value == 259
    finally:
        kernel.CloseHandle(handle)


records = []
for scenario, code, reason, script in [
    ("exit42", 42, "exited", "raise SystemExit(42)"),
    ("rss_limit", 125, "rss_limit", "import time; a=bytearray(96*1024**2); time.sleep(20)"),
    ("timeout", 124, "timeout", "import time; time.sleep(20)"),
    (
        "cancel_descendants",
        130,
        "cancelled",
        "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); print(p.pid,flush=True); time.sleep(20)",
    ),
    (
        "orphan_descendant",
        0,
        "exited",
        "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); print(p.pid,flush=True)",
    ),
]:
    log = out / (scenario + ".log")
    config = {
        "command": [sys.executable, "-u", "-c", script],
        "log_path": str(log),
        "sample_ms": 20,
        "grace_ms": 100,
        "timeout_ms": 150 if scenario == "timeout" else 4000,
        "rss_limit_bytes": 48 * 1024**2 if scenario == "rss_limit" else 256 * 1024**2,
    }
    child = subprocess.Popen(
        [str(root / "target/release/llm-supervisor.exe")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child.stdin.write(json.dumps(config) + "\n")
    child.stdin.flush()
    responses = queue.Queue()

    def reader(stream=child.stdout, target=responses):
        for line in stream:
            target.put(json.loads(line))

    threading.Thread(target=reader, daemon=True).start()
    started = time.monotonic()
    cancelled = False
    events = []
    try:
        while time.monotonic() - started < 10:
            response = responses.get(timeout=5)
            events.append(response)
            if response["event"] == "error":
                raise AssertionError(response)
            if (
                scenario == "cancel_descendants"
                and not cancelled
                and log.exists()
                and log.read_text().strip()
            ):
                child.stdin.write('{"command":"cancel"}\n')
                child.stdin.flush()
                cancelled = True
            if response["event"] == "finished":
                break
        child.stdin.close()
        assert child.wait(timeout=3) == code
        assert events[-1]["event"] == "finished" and events[-1]["reason"] == reason
        descendant_stopped = None
        if "descendant" in scenario:
            descendant_stopped = not alive(int(log.read_text().strip()))
            assert descendant_stopped
        records.append(
            {
                "scenario": scenario,
                "status": "passed",
                "descendant_stopped": descendant_stopped,
                "result": events[-1],
            }
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
result = {
    "platform": "Windows 11 Pro x86_64 MSVC",
    "rust": "1.96.1",
    "python": sys.version.split()[0],
    "scope": "Native Windows suspended-start/job assignment, RSS and deadline termination, descendant cleanup and exit42",
    "records": records,
}
(out / "windows-validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
