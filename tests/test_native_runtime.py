import os
import shutil
import sys
from pathlib import Path

import psutil
import pytest

from llm_optimise.native_runtime import NativeWorker


def binary():
    configured = os.environ.get("LLM_OPTIMISE_SUPERVISOR_BINARY")
    path = (
        Path(configured)
        if configured
        else Path(__file__).parents[1]
        / "native"
        / "target"
        / "release"
        / ("llm-supervisor.exe" if os.name == "nt" else "llm-supervisor")
    )
    if not path.is_file():
        available = shutil.which("llm-supervisor")
        if not available:
            pytest.skip("optional Rust supervisor is built in the native CI job")
        path = Path(available)
    return str(path.resolve())


def test_native_worker_starts_and_cleans_owned_process(tmp_path):
    worker = NativeWorker(
        binary(), [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "worker.log"
    )
    pid = worker.pid
    assert psutil.pid_exists(pid)
    worker.close()
    worker.close()
    assert not psutil.pid_exists(pid)
    assert worker.process.poll() is not None
    assert worker.final["reason"] in {"cancelled", "stdin_eof"}


def test_native_worker_preserves_exit_status(tmp_path):
    worker = NativeWorker(
        binary(),
        [sys.executable, "-c", "import time; time.sleep(.1); raise SystemExit(42)"],
        tmp_path / "worker.log",
    )
    worker.process.wait(timeout=5)
    worker.close()
    assert worker.final["worker_exit_code"] == 42


def test_native_worker_enforces_rss_limit(tmp_path):
    worker = NativeWorker(
        binary(),
        [sys.executable, "-c", "import time; data=bytearray(100*1024*1024); time.sleep(30)"],
        tmp_path / "worker.log",
        rss_limit_bytes=40 * 1024**2,
    )
    worker.process.wait(timeout=10)
    worker.close()
    assert worker.final["reason"] == "rss_limit"


def test_missing_native_binary_does_not_launch_fallback(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        NativeWorker(str(tmp_path / "missing"), [sys.executable, "-c", "pass"], tmp_path / "log")
