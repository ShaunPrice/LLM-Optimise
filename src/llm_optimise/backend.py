"""Managed llama.cpp and explicit OpenAI-compatible endpoints. No shell interpolation."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request

import psutil

from .network import open_request


@dataclass
class Generation:
    text: str
    latency_s: float
    ttft_s: float | None
    output_tokens: int | None
    decode_tokens_s: float | None
    prompt_tokens: int | None = None
    prompt_tokens_s: float | None = None
    truncated: bool = False
    draft_tokens: int | None = None
    accepted_draft_tokens: int | None = None
    speculation_source: str | None = None


def _probe_output(executable, option):
    # A file-backed capture bounds in-memory output from a runtime's diagnostic command.
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(
                [executable, option],
                stdout=output,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=10,
                check=False,
                env=env,
            )
            output.seek(0)
            return result.returncode, output.read(256 * 1024).decode("utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"{type(exc).__name__}: diagnostic command failed"


@lru_cache(maxsize=32)
def _runtime_capabilities(executable, mtime_ns, size):
    help_code, help_text = _probe_output(executable, "--help")
    version_code, version = _probe_output(executable, "--version")
    flags = sorted(set(re.findall(r"(?<![\w-])--[a-zA-Z][\w-]*", help_text)))
    device_code, device_text = (None, "--list-devices not advertised")
    if help_code == 0 and "--list-devices" in flags:
        device_code, device_text = _probe_output(executable, "--list-devices")
    devices = []
    patterns = {"cuda": r"cuda", "metal": r"metal|^mtl", "vulkan": r"vulkan", "rocm": r"rocm|hip"}
    if device_code == 0:
        for line in device_text.splitlines():
            match = re.match(r"\s*([\w.-]+):\s*(.+)", line)
            if not match:
                continue
            for accelerator, pattern in patterns.items():
                if re.search(pattern, match[1], re.I):
                    devices.append(
                        {"id": match[1], "accelerator": accelerator, "description": match[2]}
                    )
                    break
    cpu = help_code == 0 and all(
        f in flags for f in ("--gpu-layers", "--no-kv-offload", "--no-op-offload")
    )
    return {
        "executable": executable,
        "installed": True,
        "version": version.strip() if version_code == 0 else None,
        "help_available": help_code == 0,
        "flags": flags,
        "devices": devices,
        "supported_accelerators": (["cpu"] if cpu else [])
        + sorted({d["accelerator"] for d in devices}),
        "device_probe_exit_code": device_code,
        "device_probe": device_text.strip(),
        "evidence": "installed runtime help and device enumeration; model execution remains untested",
    }


def runtime_capabilities(executable):
    """Probe the exact installed executable; never infer support from a filename."""
    resolved = shutil.which(str(executable))
    if not resolved:
        return {
            "executable": str(executable),
            "installed": False,
            "flags": [],
            "devices": [],
            "supported_accelerators": [],
            "evidence": "executable not found",
        }
    path = Path(resolved).resolve()
    stat = path.stat()
    # Return a copy so API callers cannot mutate cached capability evidence.
    return json.loads(json.dumps(_runtime_capabilities(str(path), stat.st_mtime_ns, stat.st_size)))


def _flag(capabilities, *choices):
    if not capabilities.get("help_available"):
        raise ValueError("installed runtime help probe failed; support cannot be verified")
    for choice in choices:
        if choice in capabilities["flags"]:
            return choice
    raise ValueError("installed runtime does not advertise required flag: " + " or ".join(choices))


def speculation_counts(event):
    """Read explicit counters only; missing counters are never inferred from output length."""
    for label, values in (("terminal", event), ("timings", event.get("timings", {}))):
        if not isinstance(values, dict):
            continue
        drafted, accepted = values.get("draft_n"), values.get("draft_n_accepted")
        if type(drafted) is int and type(accepted) is int and 0 <= accepted <= drafted:
            return drafted, accepted, f"{label}.draft_n / {label}.draft_n_accepted"
    return None, None, None


def request_json(url, payload=None, timeout=30, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(
        url, data=None if payload is None else json.dumps(payload).encode(), headers=headers
    )
    with open_request(request, timeout=timeout) as response:
        return json.load(response)


def stream_json(url, payload, timeout, key, check):
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(url, data=json.dumps(payload).encode(), headers=headers)
    deadline = time.monotonic() + timeout
    with open_request(request, timeout=timeout) as response:
        data = []
        for raw in response:
            check()
            if time.monotonic() > deadline:
                raise TimeoutError("generation exceeded wall-clock timeout")
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                if data:
                    event = "\n".join(data)
                    data = []
                    if event == "[DONE]":
                        return
                    yield json.loads(event)
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data and "\n".join(data) != "[DONE]":
            yield json.loads("\n".join(data))


def llama_command(candidate, port, key):
    exe = shutil.which(candidate.executable)
    if not exe:
        raise FileNotFoundError(f"llama-server executable not found: {candidate.executable}")
    for path in (candidate.model, candidate.draft_model):
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"model file not found: {path}")
    cmd = [
        exe,
        "--model",
        candidate.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        key,
        "--ctx-size",
        str(candidate.context),
        "--threads",
        str(candidate.threads),
        "--gpu-layers",
        str(candidate.gpu_layers),
        "--batch-size",
        str(candidate.batch_size),
        "--ubatch-size",
        str(candidate.ubatch_size),
        "--cache-type-k",
        candidate.cache_type_k,
        "--cache-type-v",
        candidate.cache_type_v,
        "--flash-attn",
        candidate.flash_attention,
        "--parallel",
        "1",
    ]
    if not candidate.mmap:
        cmd.append("--no-mmap")
    if candidate.gpu_layers == 0:
        cmd.extend(["--no-kv-offload", "--no-op-offload"])
    capabilities = None
    if candidate.accelerator != "auto" or candidate.device or candidate.draft_model:
        capabilities = runtime_capabilities(exe)
    if candidate.accelerator != "auto":
        if candidate.accelerator not in capabilities["supported_accelerators"]:
            raise ValueError(f"installed runtime cannot verify accelerator {candidate.accelerator}")
    if candidate.device or candidate.accelerator not in ("auto", "cpu"):
        eligible = [
            d
            for d in capabilities["devices"]
            if candidate.accelerator in ("auto", d["accelerator"])
        ]
        device = candidate.device or (eligible[0]["id"] if eligible else None)
        if device not in {d["id"] for d in eligible}:
            raise ValueError("requested device was not enumerated for this accelerator")
        cmd.extend([_flag(capabilities, "--device"), device])
    if candidate.draft_model:
        cmd.extend(
            [_flag(capabilities, "--spec-draft-model", "--model-draft"), candidate.draft_model]
        )
        if "--spec-type" in capabilities["flags"]:
            cmd.extend(["--spec-type", "draft-simple"])
        if candidate.draft_max is not None:
            cmd.extend(
                [_flag(capabilities, "--spec-draft-n-max", "--draft-max"), str(candidate.draft_max)]
            )
        if candidate.gpu_layers == 0:
            cmd.extend([_flag(capabilities, "--spec-draft-ngl", "--gpu-layers-draft"), "0"])
            cmd.extend([_flag(capabilities, "--spec-draft-device", "--device-draft"), "none"])
    return cmd


class Backend:
    def __init__(self, candidate, log_path):
        self.candidate = candidate
        self.log_path = Path(log_path)
        self.process = None
        self.log = None
        self.endpoint = candidate.endpoint
        self.key = os.environ.get(candidate.api_key_env) if candidate.api_key_env else None
        if candidate.api_key_env and not self.key:
            raise ValueError(f"missing API key environment variable: {candidate.api_key_env}")
        self.version = None
        self.command = None
        self.started_at = None
        self._close_lock = threading.Lock()
        self._owns_process_group = False
        self.native_worker = None

    @property
    def pid(self):
        if self.native_worker:
            return self.native_worker.pid
        return self.process.pid if self.process else None

    def launch(self):
        if self.candidate.backend == "openai":
            parsed = urlparse(self.endpoint)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "endpoint must be an http(s) URL without credentials, query or fragment"
                )
            if (
                self.key
                and parsed.scheme != "https"
                and parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            ):
                raise ValueError("remote API credentials require HTTPS")
            self.endpoint = self.endpoint.rstrip("/")
            self.version = "external endpoint: record server version independently"
            return
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{port}"
        self.key = secrets.token_urlsafe(24)
        cmd = llama_command(self.candidate, port, self.key)
        self.command = ["<redacted>" if arg == self.key else arg for arg in cmd]
        version = subprocess.run(
            [cmd[0], "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        self.version = (version.stdout + version.stderr).strip() or None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = (
            {"start_new_session": True}
            if os.name != "nt"
            else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        )
        # Prevent ambient llama environment settings from silently changing an experiment.
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
        self.started_at = time.monotonic()
        if self.candidate.supervisor_executable:
            from .native_runtime import NativeWorker

            self.native_worker = NativeWorker(
                self.candidate.supervisor_executable, cmd, self.log_path, env=env
            )
            self.process = self.native_worker.process
            return
        self.log = open(self.log_path, "wb")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=env,
            **kwargs,
        )
        self._owns_process_group = os.name != "nt"

    def ready(self, timeout, check):
        if not self.process:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            check()
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited {self.process.returncode}; inspect {self.log_path.name}"
                )
            try:
                # Authenticated endpoint confirms we reached OUR server if the ephemeral port raced.
                request_json(self.endpoint + "/props", timeout=0.5, key=self.key)
                return
            except (OSError, URLError, HTTPError, ValueError):
                time.sleep(0.1)
        raise TimeoutError("llama-server startup timed out")

    def generate(self, task, max_tokens, seed, timeout, check):
        messages = [
            {"role": "system", "content": task.system},
            {"role": "user", "content": task.prompt},
        ]
        # Python <=3.12 uses a coarse Windows monotonic clock. Request metrics
        # need the performance counter to resolve fast local responses and TTFT.
        started = time.perf_counter()
        if self.candidate.backend == "llama.cpp":
            template = request_json(
                self.endpoint + "/apply-template",
                {"messages": messages},
                timeout=timeout,
                key=self.key,
            )
            payload = {
                "prompt": template["prompt"],
                "n_predict": max_tokens,
                "temperature": 0,
                "seed": seed,
                "stream": True,
                "cache_prompt": self.candidate.cache_prompt,
            }
            if self.candidate.constrain_json and task.json_schema:
                payload["json_schema"] = task.json_schema
            url = self.endpoint + "/completion"
        else:
            payload = {
                "model": self.candidate.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": seed,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if self.candidate.constrain_json and task.json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "task_output",
                        "strict": True,
                        "schema": task.json_schema,
                    },
                }
            url = self.endpoint + "/chat/completions"
        text, ttft, tokens, prompt_tokens, decode, prefill = [], None, None, None, None, None
        terminal = False
        truncated = False
        drafted, accepted, speculation_source = None, None, None
        remaining = timeout - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("prompt preparation exceeded timeout")
        for event in stream_json(url, payload, remaining, self.key, check):
            if "error" in event:
                raise RuntimeError("backend returned a streaming error")
            if self.candidate.backend == "llama.cpp":
                content = event.get("content", "")
                if event.get("stop"):
                    terminal = True
                    timings = event.get("timings", {})
                    tokens = timings.get("predicted_n", event.get("tokens_predicted"))
                    prompt_tokens = timings.get("prompt_n")
                    decode = timings.get("predicted_per_second")
                    prefill = timings.get("prompt_per_second")
                    truncated = event.get("truncated", False) or (
                        event.get("stopped_limit", False) or event.get("stop_type") == "limit"
                    )
                    drafted, accepted, speculation_source = speculation_counts(event)
            else:
                choices = event.get("choices", [])
                content = choices[0].get("delta", {}).get("content") or "" if choices else ""
                if choices and choices[0].get("finish_reason"):
                    terminal = True
                    truncated = choices[0]["finish_reason"] == "length"
                usage = event.get("usage") or {}
                tokens = usage.get("completion_tokens", tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - started
                text.append(content)
        if not terminal:
            raise RuntimeError("stream ended without a completion marker; partial output rejected")
        return Generation(
            "".join(text),
            time.perf_counter() - started,
            ttft,
            tokens,
            decode,
            prompt_tokens,
            prefill,
            truncated,
            drafted,
            accepted,
            speculation_source,
        )

    def close(self):
        # The cancellation guard and runner finally block may arrive together.
        with self._close_lock:
            if self.native_worker:
                self.native_worker.close()
                return
            if self.process:
                children = []
                try:
                    children = psutil.Process(self.process.pid).children(recursive=True)
                except (psutil.Error, OSError):
                    pass  # Still terminate our Popen child when process enumeration is restricted.
                running = self.process.poll() is None
                if running and self._owns_process_group:
                    try:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                for child in children:
                    try:
                        child.terminate()
                    except psutil.Error:
                        pass
                if running:
                    try:
                        self.process.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if self._owns_process_group:
                        try:
                            os.killpg(self.process.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                    self.process.kill()
                    self.process.wait(timeout=3)
                _, alive = psutil.wait_procs(children, timeout=1)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
            if self.log:
                self.log.close()
