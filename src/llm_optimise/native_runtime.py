"""Optional Rust supervisor protocol. No automatic fallback after an uncertain spawn."""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil


class NativeWorker:
    def __init__(
        self, executable, command, log_path, *, env=None, rss_limit_bytes=None, timeout_ms=3600000
    ):
        binary = shutil.which(str(executable))
        if not binary:
            raise ValueError(
                "native supervisor executable is unavailable; build native/ or choose Python supervision"
            )
        self.process = None
        self.pid = None
        self.metrics = {}
        self.final = None
        self._worker = None
        self._events = queue.Queue(maxsize=32)
        self._lock = threading.Lock()
        self.process = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        request = {
            "command": command,
            "cwd": str(Path.cwd()),
            "log_path": str(Path(log_path).resolve()),
            "timeout_ms": timeout_ms,
            "sample_ms": 50,
            "grace_ms": 500,
            "cancel_on_stdin_eof": True,
        }
        if rss_limit_bytes is not None:
            request["rss_limit_bytes"] = rss_limit_bytes
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    event = self._events.get(timeout=0.1)
                except queue.Empty:
                    if self.process.poll() is not None:
                        raise RuntimeError(
                            "native supervisor exited before worker startup"
                        ) from None
                    continue
                if event.get("event") == "started":
                    self.pid = event["pid"]
                    try:
                        self._worker = psutil.Process(self.pid)
                    except psutil.NoSuchProcess:
                        self._worker = None
                    return
                if event.get("event") in {"error", "finished"}:
                    raise RuntimeError("native supervisor failed before worker startup")
            raise TimeoutError("native supervisor startup timed out")
        except Exception:
            self.close()
            raise

    def _read(self):
        while True:
            line = self.process.stdout.readline(65537)
            if not line:
                break
            if len(line) > 65536:
                self.metrics = {"error": "oversized supervisor event"}
                break
            try:
                event = json.loads(line)
                kind = event.get("event")
                if kind == "sample":
                    self.metrics = event
                elif kind == "finished":
                    self.final = event
                else:
                    self._events.put_nowait(event)
            except (ValueError, AttributeError, queue.Full):
                continue

    def close(self):
        with self._lock:
            if not self.process:
                return
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"command":"cancel"}\n')
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
            try:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            # Identity-checked emergency cleanup if the supervisor itself died unexpectedly.
            if self._worker is not None:
                try:
                    if self._worker.is_running():
                        children = self._worker.children(recursive=True)
                        for process in reversed([self._worker, *children]):
                            try:
                                process.kill()
                            except psutil.Error:
                                pass
                        psutil.wait_procs([self._worker, *children], timeout=2)
                except psutil.Error:
                    pass
            self._reader.join(timeout=1)
            if self.process.stdout:
                self.process.stdout.close()
