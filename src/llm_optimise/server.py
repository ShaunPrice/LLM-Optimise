"""Loopback GUI and agent API with CSRF protection and no generated-code execution."""

from __future__ import annotations

import json
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .agent import chat_messages, parse_chat_reply, parse_models, public_models
from .config import load_experiment
from .hardware import detect_hardware
from .intelligence import Intelligence
from .managed import ManagedModels
from .report import export_report
from .routing import RoutePolicy, choose_route
from .runner import run_experiment, write_json
from .training import save_recipe, training_recipe
from .workbench import LONG_OPERATIONS, execute_workbench, workbench_status
from .workspace import apply_proposal, code_messages, prepare_proposal, project_root, read_context

STATIC = Path(__file__).parent / "static"


class App:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.proposals = {}
        self.lock = threading.RLock()
        self.local = None
        self.model_path = self.workspace / ".llm-optimise" / "models.json"
        self.models = (
            parse_models(json.loads(self.model_path.read_text()))
            if self.model_path.exists()
            else []
        )
        self.result_paths = {}
        self.benchmark_active = False
        self.intelligence = Intelligence(self.workspace)
        self.managed = ManagedModels(self)

    def _results(self):
        paths = list((self.workspace / "runs").glob("*/results.json")) + list(
            (self.workspace / "validation").glob("*/results.json")
        )
        results = []
        seen = set()
        for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
            if path.is_symlink() or not path.resolve().is_relative_to(self.workspace):
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if (
                    not isinstance(value, dict)
                    or not isinstance(value.get("name"), str)
                    or not isinstance(value.get("trials"), list)
                    or not isinstance(value.get("frontier"), list)
                ):
                    continue
                run_id = path.parent.name
                if run_id in seen:
                    continue
                seen.add(run_id)
                self.result_paths[run_id] = path
                results.append({**value, "id": run_id})
            except (OSError, ValueError):
                continue
        return results

    def hardware(self):
        hardware = detect_hardware()
        if not hardware["runtimes"]["llama-server"]:
            found = list((self.workspace / ".work" / "llama").glob("*/llama-server"))
            if found:
                hardware["runtimes"]["llama-server"] = str(found[0])
        return hardware

    def state(self):
        with self.lock:
            task_paths = list((self.workspace / "examples").glob("*.jsonl"))
            if not task_paths:
                task_paths = [Path(__file__).parent / "data" / "tasks.jsonl"]
            return {
                "hardware": self.hardware(),
                "workspace": str(self.workspace),
                "models": public_models(self.models),
                "tasks": [{"path": str(p), "label": p.stem} for p in task_paths],
                "gguf_models": [
                    str(p) for p in (self.workspace / "models").glob("*.gguf") if p.is_file()
                ],
                "jobs": [
                    {k: v for k, v in job.items() if k != "cancel"} for job in self.jobs.values()
                ],
                "results": self._results(),
                "workbench": workbench_status(self.workspace, self.intelligence),
                "lifecycle": self.managed.status(),
                "local_server": {
                    "status": "running",
                    "endpoint": self.local.endpoint,
                    "model": self.local.candidate.model,
                }
                if self.local and self.local.process and self.local.process.poll() is None
                else {"status": "stopped"},
                "projects": sorted(
                    p.name
                    for p in (self.workspace / "projects").glob("*")
                    if p.is_dir() and not p.is_symlink()
                ),
            }

    def start_job(self, kind, operation):
        with self.lock:
            active = [j for j in self.jobs.values() if j["status"] == "running"]
            concurrent = {"chat", "code", "specialist", "context"}
            if active and (
                kind not in concurrent
                or any(j["kind"] not in concurrent for j in active)
                or len(active) >= 4
            ):
                raise ValueError("another operation is running; finish or cancel it first")
            job_id = uuid.uuid4().hex[:12]
            job = {
                "id": job_id,
                "kind": kind,
                "status": "running",
                "progress": {"phase": "starting"},
                "cancel": threading.Event(),
            }
            self.jobs[job_id] = job
            if len(self.jobs) > 40:
                finished = next(
                    (key for key, value in self.jobs.items() if value["status"] != "running"), None
                )
                if finished is not None:
                    del self.jobs[finished]

        def worker():
            try:
                result = operation(job)
                with self.lock:
                    job["result"] = result
                    job["status"] = "cancelled" if job["cancel"].is_set() else "complete"
            except Exception as exc:
                with self.lock:
                    job["status"] = "failed"
                    job["error"] = str(exc)
            finally:
                record = {k: v for k, v in job.items() if k != "cancel"}
                write_json(self.workspace / "runs" / "jobs" / f"{job_id}.json", record)

        threading.Thread(target=worker, daemon=True).start()
        return {"id": job_id}

    def experiment(self, config):
        with self.lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise ValueError("finish or cancel the active operation before benchmarking")
            self.managed.unload()
        name = "run-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        config_path = self.workspace / ".llm-optimise" / "experiments" / f"{name}.json"
        # GUI paths resolve against the workspace, not the stored config's nested directory.
        config = json.loads(json.dumps(config))
        config["dataset"] = str((self.workspace / config["dataset"]).resolve())
        for candidate in config.get("candidates", []):
            if candidate.get("backend", "llama.cpp") == "llama.cpp":
                for key in ("model", "draft_model", "supervisor_executable"):
                    if candidate.get(key):
                        candidate[key] = str((self.workspace / candidate[key]).resolve())
                exe = candidate.get("executable", "llama-server")
                if "/" in exe or "\\" in exe:
                    candidate["executable"] = str((self.workspace / exe).resolve())
        write_json(config_path, config)
        experiment = load_experiment(config_path)
        from .quality import load_tasks

        load_tasks(experiment.dataset)

        def operation(job):
            def progress(event):
                with self.lock:
                    job["progress"] = event

            path = self.workspace / "runs" / name
            result = run_experiment(experiment, path, cancel=job["cancel"], progress=progress)
            export_report(result, path)
            return {"run_id": name, **result}

        return self.start_job("experiment", operation)

    def start_local(self, data):
        return self.start_job("model", lambda job: self.managed.start(data, job["cancel"]))

    def stop_local(self):
        with self.lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise ValueError("finish or cancel the active operation before stopping the model")
            return self.managed.unload(remove=True)

    def project(self, name):
        root = project_root(self.workspace, name)
        files = []
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and not path.is_symlink()
                and not any(p.startswith(".") for p in relative.parts)
            ):
                files.append(relative.as_posix())
                if len(files) == 500:
                    break
        return {"name": name, "files": sorted(files)}

    def code(self, data):
        root = project_root(self.workspace, data["project"])
        context = read_context(root, data.get("context_files", []))
        selection = None
        if data.get("context_selection") is not None:
            from .context import select_context

            selection = select_context(data["prompt"], context, **data["context_selection"])
            if not selection["required_fact_gate"]:
                raise ValueError(
                    "selected context omits required facts; increase the budget or use full context"
                )
        messages = code_messages(
            data["prompt"], {"selected-context.txt": selection["text"]} if selection else context
        )
        models = list(self.models)

        def operation(job):
            result = self.intelligence.run(
                models,
                messages,
                data.get("policy"),
                data.get("selected_model"),
                data.get("max_tokens", 2048),
                "code",
                **data.get("optimisation", {}),
                cancel=job["cancel"],
                lease=lambda model: self.managed.lease(model, job["cancel"]),
            )
            if job["cancel"].is_set():
                return {"cancelled": True}
            proposal = prepare_proposal(root, result["completion"]["text"], context)
            proposal_id = uuid.uuid4().hex
            with self.lock:
                self.proposals[proposal_id] = (root, proposal)
            return {
                **result,
                "proposal_id": proposal_id,
                "proposal": proposal,
                "context_selection": selection,
            }

        return self.start_job("code", operation)

    def chat(self, data):
        state = self.state()
        context = {k: state[k] for k in ("hardware", "tasks", "gguf_models")}
        context["results"] = [
            {
                "name": r["name"],
                "frontier": r["frontier"],
                "trials": [
                    {k: t.get(k) for k in ("name", "metrics", "memory", "rejection_reasons")}
                    for t in r["trials"][:12]
                ],
            }
            for r in state["results"][:2]
        ]
        messages = chat_messages(data["message"], data.get("history", []), context)
        models = list(self.models)

        def operation(job):
            result = self.intelligence.run(
                models,
                messages,
                data.get("policy"),
                data.get("selected_model"),
                data.get("max_tokens", 1024),
                "chat",
                **data.get("optimisation", {}),
                cancel=job["cancel"],
                lease=lambda model: self.managed.lease(model, job["cancel"]),
            )
            parsed = parse_chat_reply(result["completion"]["text"])
            if parsed["experiment"] is not None:
                try:
                    self.validate_suggestion(parsed["experiment"], state)
                except (ValueError, TypeError, KeyError) as exc:
                    parsed["experiment"] = None
                    parsed["reply"] += f"\n\nExperiment suggestion rejected: {exc}"
            return {**result, **parsed}

        return self.start_job("chat", operation)

    def validate_suggestion(self, config, state):
        if not isinstance(config, dict) or config.get("dataset") not in [
            t["path"] for t in state["tasks"]
        ]:
            raise ValueError("use an available task dataset")
        known_models = set(state["gguf_models"])
        known_exe = {"llama-server", state["hardware"]["runtimes"].get("llama-server")}
        for candidate in config.get("candidates", []):
            if (
                candidate.get("backend", "llama.cpp") != "llama.cpp"
                or candidate.get("model") not in known_models
                or candidate.get("executable", "llama-server") not in known_exe
            ):
                raise ValueError("suggestions may only use known local models and runtime")
            if candidate.get("draft_model") or set(candidate.get("sweep", {})) - {
                "threads",
                "context",
                "gpu_layers",
                "batch_size",
                "ubatch_size",
                "cache_type_k",
                "cache_type_v",
                "flash_attention",
            }:
                raise ValueError("unsupported suggestion settings")
        if (
            config.get("max_trials", 32) > 8
            or config.get("repeats", 3) > 3
            or config.get("max_tokens", 64) > 128
        ):
            raise ValueError(
                "suggested experiments are limited to 8 trials, 3 repeats and 128 output tokens"
            )

    def mutate(self, path, data):
        if path.startswith("/api/workbench/"):
            operation = path.rsplit("/", 1)[-1]

            def work(job):
                def progress(event):
                    with self.lock:
                        job["progress"] = event

                result = execute_workbench(
                    self.workspace,
                    operation,
                    data,
                    models=list(self.models),
                    engine=self.intelligence,
                    output_dir=self.workspace / "runs" / "workbench" / job["id"],
                    cancel=job["cancel"],
                    progress=progress,
                    lease=lambda model: self.managed.lease(model, job["cancel"]),
                )
                if operation == "feedback" and result.get("proposal"):
                    proposal_id = uuid.uuid4().hex
                    with self.lock:
                        self.proposals[proposal_id] = (
                            project_root(self.workspace, data["project"]),
                            result["proposal"],
                        )
                    result["proposal_id"] = proposal_id
                return result

            if operation in LONG_OPERATIONS:
                if operation in {
                    "explore-run",
                    "training-run",
                    "adapter-evaluate",
                    "adapter-reload",
                    "component",
                }:
                    with self.lock:
                        if any(j["status"] == "running" for j in self.jobs.values()):
                            raise ValueError(
                                "finish or cancel active work before a resource experiment"
                            )
                        self.managed.unload()
                return self.start_job(operation, work)
            return work({"id": uuid.uuid4().hex[:12], "cancel": threading.Event()})
        if path == "/api/lifecycle/settings":
            return self.managed.configure(data)
        if path == "/api/lifecycle/unload":
            return self.managed.unload(data.get("id"), remove=data.get("remove", False))
        if path == "/api/experiment":
            return self.experiment(data["config"])
        if path == "/api/cancel":
            with self.lock:
                job = self.jobs[data["id"]]
                job["cancel"].set()
            return {
                "status": "cancellation requested; external calls may finish before cancellation"
            }
        if path == "/api/models":
            models = parse_models(data["models"])
            models = [m for m in models if m.id not in self.managed.descriptors]
            write_json(self.model_path, public_models(models))
            with self.lock:
                self.models = models + [m for m in self.models if m.id in self.managed.descriptors]
            return {"models": public_models(self.models)}
        if path == "/api/route":
            models = self.models
            evidence = {}
            if data.get("calibrated", False):
                models, evidence = self.intelligence.calibrated_models(
                    models, data.get("task_class", "general"), data.get("input_tokens", 1000)
                )
            return {
                **choose_route(
                    models,
                    RoutePolicy(**data.get("policy", {})),
                    data.get("input_tokens", 1000),
                    data.get("output_tokens", 256),
                    data.get("capability", "code"),
                    data.get("selected_model"),
                ),
                "calibration": evidence,
            }
        if path == "/api/project":
            return self.project(data["name"])
        if path == "/api/code":
            return self.code(data)
        if path == "/api/chat":
            return self.chat(data)
        if path == "/api/apply":
            with self.lock:
                root, proposal = self.proposals[data["proposal_id"]]
                return {"files": apply_proposal(root, proposal)}
        if path == "/api/local/start":
            return self.start_local(data)
        if path == "/api/local/stop":
            return self.stop_local()
        if path == "/api/container":
            from .containers import run_container

            root = project_root(self.workspace, data["project"])
            options = {k: v for k, v in data.items() if k != "project"}
            if set(options) - {
                "runtime",
                "action",
                "memory_mib",
                "cpus",
                "timeout_s",
                "network",
                "pull",
            }:
                raise ValueError("unsupported container option")

            def operation(job):
                return run_container(
                    root,
                    self.workspace / "runs" / "containers" / job["id"],
                    cancel=job["cancel"],
                    **options,
                )

            return self.start_job("container", operation)
        if path == "/api/training":
            recipe = training_recipe(
                data["engine"],
                data["model"],
                data["data"],
                data["output"],
                max_length=data.get("max_length", 512),
                rank=data.get("rank", 8),
            )
            return save_recipe(
                recipe, self.workspace / "runs" / "recipes" / f"soup-{uuid.uuid4().hex[:8]}.yaml"
            )
        raise KeyError("unknown API endpoint")


def make_server(workspace, port=8765, host="127.0.0.1"):
    app = App(workspace)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def trusted(self, mutation=False):
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in hosts:
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in hosts}:
                return False
            return not mutation or secrets.compare_digest(
                self.headers.get("X-LLM-Token", ""), app.token
            )

        def respond(self, data, status=200, content_type="application/json"):
            body = (
                json.dumps(data, allow_nan=False).encode()
                if content_type == "application/json"
                else data
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.trusted():
                return self.respond({"error": "untrusted request origin or Host"}, 403)
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/state":
                    return self.respond(app.state())
                if parsed.path == "/api/project":
                    return self.respond(app.project(parse_qs(parsed.query)["name"][0]))
                if parsed.path.startswith(("/api/results/", "/api/export/")):
                    run_id = parsed.path.rsplit("/", 1)[-1]
                    app._results()
                    return self.respond(
                        json.loads(app.result_paths[run_id].read_text(encoding="utf-8"))
                    )
                files = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                    "/app.css": ("app.css", "text/css; charset=utf-8"),
                    "/app-icon.png": ("app-icon.png", "image/png"),
                    "/workbench.js": ("workbench.js", "text/javascript; charset=utf-8"),
                    "/workbench.css": ("workbench.css", "text/css; charset=utf-8"),
                }
                name, mime = files[parsed.path]
                body = (STATIC / name).read_bytes()
                if name == "index.html":
                    body = body.replace(b"__CSRF_TOKEN__", app.token.encode())
                return self.respond(body, content_type=mime)
            except (KeyError, FileNotFoundError):
                return self.respond({"error": "not found"}, 404)
            except (ValueError, TypeError, OSError) as exc:
                return self.respond({"error": str(exc)}, 400)

        def do_POST(self):
            if not self.trusted(mutation=True):
                return self.respond({"error": "invalid CSRF token or request origin"}, 403)
            try:
                size = int(self.headers.get("Content-Length", 0))
                if size < 1 or size > 2 * 1024 * 1024:
                    raise ValueError("request body must be 1 byte–2 MiB")
                self.connection.settimeout(15)
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("request must be a JSON object")
                return self.respond(app.mutate(urlparse(self.path).path, data))
            except (ValueError, TypeError, KeyError, OSError) as exc:
                return self.respond({"error": str(exc)}, 400)
            except Exception:
                return self.respond({"error": "operation failed; inspect local terminal"}, 500)

    class LabServer(ThreadingHTTPServer):
        def server_close(self):
            for job in self.app.jobs.values():
                job["cancel"].set()
            self.app.managed.close()
            super().server_close()

    server = LabServer((host, port), Handler)
    server.app = app
    return server


def serve(workspace, port=8765, open_browser=False, host="127.0.0.1"):
    server = make_server(workspace, port, host)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"LLM-Optimise: {url}", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in server.app.jobs.values():
            job["cancel"].set()
        if server.app.local:
            server.app.local.close()
        server.server_close()
