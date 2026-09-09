"""Connect model residency leases to the application's provider registry."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
from contextlib import contextmanager
from pathlib import Path

from .backend import Backend
from .config import Candidate
from .intelligence import fingerprint
from .lifecycle import ModelLifecycleManager, sample_headroom
from .routing import ProviderModel


class ManagedModels:
    def __init__(self, app):
        self.app = app
        self.descriptors = {}
        self._manager = None
        self.settings = {
            "max_active_leases": 2,
            "max_models": 2,
            "min_ram_gib": 1,
            "min_gpu_gib": 0.25,
            "idle_timeout_s": 60,
        }

    @property
    def manager(self):
        if self._manager is None:
            self._manager = ModelLifecycleManager(**self.settings)
        return self._manager

    def configure(self, settings):
        if set(settings) - set(self.settings):
            raise ValueError("unsupported lifecycle setting")
        if self.descriptors:
            raise ValueError("remove managed models before changing lifecycle settings")
        candidate = ModelLifecycleManager(**(self.settings | settings), start_reaper=False)
        candidate.close()
        if self._manager:
            self._manager.close()
        self._manager = None
        self.settings.update(settings)
        return self.status()

    def status(self):
        status = self._manager.status() if self._manager else {"models": [], "active_leases": 0}
        return {
            **status,
            "settings": self.settings,
            "registered": [
                {
                    "id": key,
                    "model": value["candidate"].model,
                    "context": value["candidate"].context,
                    "requirements": value["requirements"],
                }
                for key, value in self.descriptors.items()
            ],
        }

    def start(self, data, cancel=None):
        model_id = data.get("id", "managed-local")
        if not isinstance(model_id, str) or not re.fullmatch(
            r"managed-[a-zA-Z0-9_-]{1,48}", model_id
        ):
            raise ValueError(
                "managed model ID must start with managed- and use letters, numbers or dashes"
            )
        model_path = (self.app.workspace / data["model"]).resolve()
        if not model_path.is_file():
            raise ValueError("download a local GGUF model before starting it")
        values = {
            k: data[k]
            for k in (
                "threads",
                "context",
                "gpu_layers",
                "batch_size",
                "ubatch_size",
                "cache_type_k",
                "cache_type_v",
                "flash_attention",
                "accelerator",
                "device",
                "supervisor_executable",
            )
            if k in data
        }
        exe = data.get("executable", "llama-server")
        if "/" in exe or "\\" in exe:
            exe = str((self.app.workspace / exe).resolve())
        candidate = Candidate(name=model_id, model=str(model_path), executable=exe, **values)
        stat = model_path.stat()
        key = fingerprint(
            {
                "candidate": {
                    k: v for k, v in dataclasses.asdict(candidate).items() if k != "name"
                },
                "size": stat.st_size,
                "mtime": stat.st_mtime_ns,
            }
        )
        if model_id in self.descriptors and self.descriptors[model_id]["key"] != key:
            raise ValueError(
                "remove this model registration before changing its runtime configuration"
            )
        digest = hashlib.sha256()
        with model_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("model registration cancelled")
                digest.update(chunk)
        headroom = sample_headroom()
        estimate = stat.st_size / 1024**3 * 1.25 + 0.5 + candidate.context * 2048 / 1024**3
        gpu = (
            0
            if candidate.gpu_layers == 0 or headroom["unified_memory"]
            else data.get("gpu_gib", estimate)
        )
        requirements = {
            "ram_gib": data.get("ram_gib", estimate),
            "gpu_gib": gpu,
            "working_ram_gib": data.get("working_ram_gib", 0.125),
            "max_concurrency": data.get("max_concurrency", 1),
            "max_rss_gib": data.get("max_rss_gib"),
            "max_gpu_gib": data.get("max_gpu_gib"),
        }
        descriptor = {
            "key": key,
            "candidate": candidate,
            "requirements": requirements,
            "revision": digest.hexdigest(),
            "stat": (stat.st_size, stat.st_mtime_ns),
            "backend": None,
        }
        env_name = "LLM_OPTIMISE_LOCAL_" + fingerprint(model_id)[:16].upper()

        def factory():
            backend = Backend(candidate, self.app.workspace / "runs" / f"model-{key[:12]}.log")
            descriptor["backend"] = backend
            return backend

        descriptor["factory"] = factory
        descriptor["env_name"] = env_name
        with self.manager.acquire(key, factory, cancel=cancel, **requirements) as lease:
            backend = lease.backend
            descriptor["backend"] = backend
            os.environ[env_name] = backend.key
            provider = ProviderModel(
                id=model_id,
                model=model_path.name,
                provider="openai",
                location="local",
                base_url=backend.endpoint + "/v1",
                context_window=candidate.context,
                max_output_tokens=min(2048, candidate.context // 2),
                api_key_env=env_name,
                input_cost_per_million=0,
                output_cost_per_million=0,
                supports_json_schema=True,
                revision=digest.hexdigest(),
                ram_gib=requirements["ram_gib"],
                gpu_gib=gpu,
            )
            with self.app.lock:
                self.descriptors[model_id] = descriptor
                self.app.models = [m for m in self.app.models if m.id != model_id] + [provider]
                self.app.local = backend
        return {
            "model_id": model_id,
            "endpoint": backend.endpoint,
            "revision": digest.hexdigest(),
            "requirements": requirements,
            "admission_note": "Weight/context formula is an estimate unless overridden; live free-memory admission and optional sampled process caps remain separate.",
        }

    @contextmanager
    def lease(self, model, cancel=None):
        descriptor = self.descriptors.get(model.id)
        if descriptor is None:
            yield model
            return
        stat = Path(descriptor["candidate"].model).stat()
        if (stat.st_size, stat.st_mtime_ns) != descriptor["stat"]:
            raise ValueError("managed model file changed; remove and register its new revision")
        with self.manager.acquire(
            descriptor["key"], descriptor["factory"], cancel=cancel, **descriptor["requirements"]
        ) as active:
            backend = active.backend
            os.environ[descriptor["env_name"]] = backend.key
            live = dataclasses.replace(model, base_url=backend.endpoint + "/v1")
            with self.app.lock:
                self.app.models = [live if m.id == model.id else m for m in self.app.models]
                self.app.local = backend
            yield live
            active.check()

    def unload(self, model_id=None, *, remove=False):
        ids = [model_id] if model_id else list(self.descriptors)
        for name in ids:
            descriptor = self.descriptors[name]
            if self._manager:
                self._manager.unload(descriptor["key"])
            if remove:
                os.environ.pop(descriptor["env_name"], None)
                del self.descriptors[name]
                self.app.models = [m for m in self.app.models if m.id != name]
        self.app.local = None
        return {"status": "removed" if remove else "unloaded", "models": ids}

    def close(self):
        if self._manager:
            self._manager.close()
        for descriptor in self.descriptors.values():
            os.environ.pop(descriptor["env_name"], None)
