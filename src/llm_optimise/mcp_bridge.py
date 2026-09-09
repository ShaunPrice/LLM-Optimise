"""Bounded MCP access to one existing GUI session; never owns another App instance."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from jsonschema import Draft202012Validator

from .config import file_digest, parse_experiment
from .mcp_catalog import BY_ID, CATALOG

MAX_BODY = 2 * 1024**2
MAX_RESPONSE = 16 * 1024**2


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError("App redirects are refused")


class AppBridge:
    def __init__(self, workspace, app_url, *, runtimes=(), read_roots=(), max_cost_usd=1.0):
        self.workspace = Path(workspace).resolve()
        parsed = urlparse(app_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or not parsed.port
            or parsed.path not in ("", "/")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("app-url must be http://127.0.0.1:PORT with no credentials or path")
        self.url = app_url.rstrip("/")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.read_roots = [self.workspace, *(Path(p).resolve() for p in read_roots)]
        self.runtimes = {}
        for value in runtimes:
            path = self._executable_path(value)
            self.runtimes[str(path)] = file_digest(path)
        if not 0 <= max_cost_usd <= 1000:
            raise ValueError("operator max-cost-usd must be 0–1000")
        self.max_cost_usd = max_cost_usd
        self.token = None
        self.lock = threading.RLock()

    def _request(self, path, data=None, *, text=False):
        body = None if data is None else json.dumps(data, allow_nan=False).encode()
        if body is not None and len(body) > MAX_BODY:
            raise ValueError("request exceeds 2 MiB")
        headers = {"Origin": self.url}
        if data is not None:
            headers.update({"Content-Type": "application/json", "X-LLM-Token": self.token or ""})
        try:
            with self.opener.open(
                Request(self.url + path, data=body, headers=headers), timeout=60
            ) as response:
                raw = response.read(MAX_RESPONSE + 1)
        except HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise ValueError(f"App rejected request ({exc.code}): {self.redact(detail)}") from None
        except OSError as exc:
            raise ValueError(
                "Cannot reach the existing loopback App; start llm-optimise ui first"
            ) from exc
        if len(raw) > MAX_RESPONSE:
            raise ValueError(
                "App response exceeds 16 MiB; inspect the local GUI or smaller artifact"
            )
        return raw.decode("utf-8") if text else json.loads(raw)

    def state(self):
        state = self._request("/api/state")
        if Path(state["workspace"]).resolve() != self.workspace:
            raise ValueError(
                "App workspace differs from MCP workspace; connect to the intended session"
            )
        return state

    def _connect(self):
        state = self.state()
        if self.token is None:
            html = self._request("/", text=True)
            match = re.search(r'<meta name="csrf-token" content="([a-zA-Z0-9_-]+)"', html)
            if not match:
                raise ValueError("App did not provide a session CSRF token")
            self.token = match.group(1)
        return state

    def redact(self, value):
        if isinstance(value, dict):
            return {
                k: "<redacted>"
                if k.lower()
                in {
                    "api_key",
                    "authorization",
                    "password",
                    "secret",
                    "access_token",
                    "refresh_token",
                    "token",
                }
                else self.redact(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, str):
            if self.token:
                value = value.replace(self.token, "<redacted>")
            # These values are never intentionally exposed; redact accidental error/log echoes.
            for name, secret in os.environ.items():
                if len(secret) >= 8 and any(
                    word in name.upper() for word in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")
                ):
                    value = value.replace(secret, "<redacted>")
        return value

    def bounded(self, value, limit=24000):
        value = self.redact(value)
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded) <= limit:
            return value
        return {
            "truncated": True,
            "total_characters": len(encoded),
            "preview": encoded[:limit],
            "next_step": "Use job_result or artifact_read for bounded pages; previews are not complete JSON.",
        }

    def discover(self, query="", action_id=None, limit=6):
        if not 1 <= limit <= 12:
            raise ValueError("limit must be 1–12")
        if action_id is not None:
            if action_id not in BY_ID:
                raise ValueError("unknown action ID; use discover with keywords")
            items = [BY_ID[action_id]]
        else:
            terms = re.findall(r"[a-z0-9]+", query.lower())
            ranked = [
                (
                    sum(term in (item["id"] + " " + item["description"]).lower() for term in terms),
                    item,
                )
                for item in CATALOG
            ]
            items = [
                item
                for score, item in sorted(ranked, key=lambda pair: -pair[0])
                if score or not terms
            ][:limit]
        return {
            "actions": [{k: v for k, v in item.items() if k != "endpoint"} for item in items],
            "total_actions": len(CATALOG),
            "all_action_ids": list(BY_ID) if not query and action_id is None else None,
            "execution": "read actions use read_action; mutations use execute_action; GUI job IDs use job_status/job_result/cancel_job",
        }

    def read_path(self, value):
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("path must be nonempty")
        path = (self.workspace / value).resolve()
        for root in sorted(self.read_roots, key=lambda p: len(p.parts), reverse=True):
            if path.is_relative_to(root) and not any(
                part.startswith(".") for part in path.relative_to(root).parts
            ):
                if not path.exists():
                    raise ValueError("input path does not exist")
                if not (path.is_file() or path.is_dir()):
                    raise ValueError("input must be a regular file or directory")
                return str(path)
        raise ValueError("path leaves allowed data roots or accesses a hidden path")

    def _executable_path(self, value):
        found = shutil.which(value) if not ("/" in value or "\\" in value) else value
        if not found:
            raise ValueError("runtime executable not found")
        # Preserve venv symlinks: their spelling determines the Python environment.
        path = Path(os.path.abspath(self.workspace / found))
        if not path.is_file():
            raise ValueError("runtime executable not found")
        return path

    def runtime(self, value):
        path = self._executable_path(value)
        expected = self.runtimes.get(str(path))
        if expected is None or expected != file_digest(path):
            raise ValueError(
                "runtime is not operator-allowlisted or changed; configure --runtime before connecting"
            )
        return str(path)

    def _experiment(self, config):
        config["dataset"] = self.read_path(config["dataset"])
        parsed = parse_experiment(config, self.workspace)
        # Validate caller spellings before accepting the parser's resolved binary paths.
        # Python venv symlinks sharing an inode must not widen the operator allowlist.
        approved_runtimes = {}
        for entry in config["candidates"]:
            values = [entry.get("executable", "llama-server")]
            if entry.get("supervisor_executable"):
                values.append(entry["supervisor_executable"])
            values.extend(entry.get("sweep", {}).get("supervisor_executable", []))
            for value in values:
                if value is not None:
                    approved = self.runtime(value)
                    approved_runtimes[str(Path(approved).resolve())] = approved
        if (
            len(parsed.candidates) > 64
            or parsed.repeats > 10
            or parsed.max_tokens > 2048
            or max(parsed.timeout_s, parsed.startup_timeout_s) > 300
        ):
            raise ValueError(
                "MCP experiments allow at most 64 trials, 10 repeats, 2048 output tokens and 300-second request/startup bounds"
            )
        if parsed.limits.max_rss_gib is None or parsed.limits.min_available_gib <= 0:
            raise ValueError(
                "MCP experiments require explicit max_rss_gib and positive min_available_gib"
            )
        from .quality import load_tasks

        if (
            len(parsed.candidates)
            * (parsed.warmup + parsed.repeats * len(load_tasks(parsed.dataset)))
            > 10000
        ):
            raise ValueError("MCP experiment exceeds 10000 bounded requests")
        import dataclasses

        candidates = []
        for candidate in parsed.candidates:
            if candidate.backend != "llama.cpp":
                raise ValueError(
                    "MCP experiments use managed local workers; use calibrate for configured providers"
                )
            row = dataclasses.asdict(candidate)
            row["model"] = self.read_path(row["model"])
            row["executable"] = approved_runtimes[
                str(self._executable_path(row["executable"]).resolve())
            ]
            for key in ("draft_model", "supervisor_executable"):
                if row.get(key):
                    row[key] = (
                        approved_runtimes[str(self._executable_path(row[key]).resolve())]
                        if key == "supervisor_executable"
                        else self.read_path(row[key])
                    )
            candidates.append(row)
        config["candidates"] = candidates

    def _validate(self, action, data, state):
        Draft202012Validator(BY_ID[action]["input_schema"]).validate(data)
        # Avoid bypassing the fixed Docker command surface through nested extra fields.
        if action == "container":
            data["network"] = False
        if action == "experiment":
            self._experiment(data["config"])
        if action.startswith("explore-"):
            self._experiment(data["experiment"])
            for runtime in data.get("accelerators", []):
                runtime["executable"] = self.runtime(runtime["executable"])
            if data.get("search", {}).get("heldout_dataset"):
                data["search"]["heldout_dataset"] = self.read_path(
                    data["search"]["heldout_dataset"]
                )
            if data.get("speculative", {}).get("draft_models"):
                data["speculative"]["draft_models"] = [
                    self.read_path(path) for path in data["speculative"]["draft_models"]
                ]
        for key in ("source", "dataset", "path", "base_model"):
            if key in data:
                data[key] = self.read_path(data[key])
        if "python" in data:
            data["python"] = self.runtime(data["python"])
        if action == "training-run":
            config = data["config"]
            config["base"] = self.read_path(config["base"])
            for key in ("train", "val"):
                if config.get("data", {}).get(key):
                    config["data"][key] = self.read_path(config["data"][key])
        if action == "model-start":
            data["model"] = self.read_path(data["model"])
            data["executable"] = self.runtime(data["executable"])
            if data.get("supervisor_executable"):
                data["supervisor_executable"] = self.runtime(data["supervisor_executable"])
        if action == "models-configure":
            bindings = {
                (row.get("api_key_env"), row.get("base_url"))
                for row in state["models"]
                if row.get("api_key_env")
            }
            for row in data["models"]:
                if (
                    row.get("api_key_env")
                    and (row["api_key_env"], row.get("base_url")) not in bindings
                ):
                    raise ValueError(
                        "configure new credential/endpoint bindings in the trusted local GUI first"
                    )
        if action in {"chat", "code", "specialist", "distill", "calibrate", "feedback"}:
            policy = data.setdefault("policy", {"placement": "local", "objective": "cost"})
            budget = data.get("max_cost_usd", policy.get("max_cost_usd"))
            if (
                not isinstance(budget, (int, float))
                or isinstance(budget, bool)
                or not 0 <= budget <= self.max_cost_usd
            ):
                raise ValueError(
                    "provide an explicit max_cost_usd within the operator's per-operation cap"
                )
            policy_budget = policy.get("max_cost_usd")
            policy["max_cost_usd"] = (
                min(policy_budget, budget) if policy_budget is not None else budget
            )

    def call(self, action, arguments, *, read):
        if action not in BY_ID or BY_ID[action]["read_only"] is not read:
            raise ValueError(
                "action is unknown or belongs to the other read/write tool; use discover"
            )
        data = json.loads(json.dumps(arguments, allow_nan=False))
        with self.lock:
            state = self._connect()
            self._validate(action, data, state)
            if read:
                if action == "status":
                    return self.bounded(
                        {
                            **state,
                            "jobs": [
                                {k: v for k, v in job.items() if k != "result"}
                                for job in state["jobs"]
                            ],
                            "results": [
                                {k: row.get(k) for k in ("id", "name", "frontier", "complete")}
                                for row in state["results"]
                            ],
                        }
                    )
                if action in {"models", "lifecycle"}:
                    return self.bounded({action: state[action]})
                if action == "workbench-status":
                    return self.bounded(state["workbench"])
                if action == "results":
                    return self.bounded(
                        {
                            "results": [
                                {k: row.get(k) for k in ("id", "name", "frontier", "complete")}
                                for row in state["results"]
                            ]
                        }
                    )
                if action == "result":
                    return self.bounded(self._request("/api/results/" + data["id"]))
            return self.bounded(self._request(BY_ID[action]["endpoint"], data))

    def job(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{12}", job_id):
            raise ValueError("invalid GUI job ID")
        state = self.state()
        for job in state["jobs"]:
            if job["id"] == job_id:
                return job
        path = self.workspace / "runs" / "jobs" / (job_id + ".json")
        if (
            path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(self.workspace)
            and path.stat().st_size <= MAX_RESPONSE
        ):
            return json.loads(path.read_text())
        raise ValueError("job not found in the active session or persisted job records")

    def cancel(self, job_id):
        with self.lock:
            self._connect()
            job = self.job(job_id)
            if job["status"] != "running":
                return {"id": job_id, "status": job["status"], "cancellation_requested": False}
            self._request("/api/cancel", {"id": job_id})
            return {
                "id": job_id,
                "cancellation_requested": True,
                "note": "Cancellation is cooperative; external requests may finish. Poll job_status for terminal state.",
            }

    def artifact(self, path, offset=0, max_bytes=16384):
        if (
            type(offset) is not int
            or offset < 0
            or type(max_bytes) is not int
            or not 1 <= max_bytes <= 32768
        ):
            raise ValueError("offset must be nonnegative and max_bytes must be 1–32768")
        self.state()  # Ensure the bridge still addresses the selected workspace.
        target = Path(self.read_path(path))
        relative = (
            target.relative_to(self.workspace) if target.is_relative_to(self.workspace) else None
        )
        if (
            relative is None
            or relative.parts[0] not in {"runs", "validation"}
            or target.suffix.lower()
            not in {".json", ".jsonl", ".csv", ".md", ".txt", ".log", ".yaml"}
            or not target.is_file()
        ):
            raise ValueError(
                "artifact reads are limited to text reports below runs/ or validation/"
            )
        file_limit = 8 * 1024**2
        if target.stat().st_size > file_limit:
            raise ValueError("artifact exceeds 8 MiB; inspect it locally")
        with target.open("rb") as stream:
            raw = stream.read(file_limit + 1)
        if len(raw) > file_limit:
            raise ValueError("artifact exceeds 8 MiB; inspect it locally")
        # Redact before slicing: otherwise a secret straddling pages can be reconstructed.
        encoded = self.redact(raw.decode("utf-8", errors="replace")).encode("utf-8")
        if offset > len(encoded) or (offset < len(encoded) and encoded[offset] & 0xC0 == 0x80):
            raise ValueError("offset must be a UTF-8 boundary in the redacted artifact")
        end = min(len(encoded), offset + max_bytes)
        while end > offset and end < len(encoded) and encoded[end] & 0xC0 == 0x80:
            end -= 1
        if end == offset and offset < len(encoded):
            raise ValueError("max_bytes is too small for the next UTF-8 character; use at least 4")
        return {
            "path": relative.as_posix(),
            "offset": offset,
            "next_offset": end,
            "total_bytes": len(encoded),
            "offset_basis": "redacted UTF-8 bytes",
            "eof": end >= len(encoded),
            "text": encoded[offset:end].decode("utf-8"),
        }
