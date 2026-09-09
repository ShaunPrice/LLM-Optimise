"""Managed llama.cpp and explicit OpenAI-compatible endpoints. No shell interpolation."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
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
    if candidate.draft_model:
        cmd.extend(["--model-draft", candidate.draft_model])
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

    @property
    def pid(self):
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
        self.log = open(self.log_path, "wb")
        kwargs = (
            {"start_new_session": True}
            if os.name != "nt"
            else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        )
        # Prevent ambient llama environment settings from silently changing an experiment.
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
        self.started_at = time.monotonic()
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=env,
            **kwargs,
        )

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
        started = time.monotonic()
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
        remaining = timeout - (time.monotonic() - started)
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
                    ttft = time.monotonic() - started
                text.append(content)
        if not terminal:
            raise RuntimeError("stream ended without a completion marker; partial output rejected")
        return Generation(
            "".join(text),
            time.monotonic() - started,
            ttft,
            tokens,
            decode,
            prompt_tokens,
            prefill,
            truncated,
        )

    def close(self):
        if self.process:
            try:
                root = psutil.Process(self.process.pid)
                children = root.children(recursive=True)
                for p in children:
                    try:
                        p.terminate()
                    except psutil.NoSuchProcess:
                        pass
                if self.process.poll() is None:
                    self.process.terminate()
                _, alive = psutil.wait_procs(children, timeout=2)
                for p in alive:
                    p.kill()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            except psutil.NoSuchProcess:
                pass
        if self.log:
            self.log.close()
