"""Strict, portable experiment specifications. Relative paths follow the config file."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    backend: str = "llama.cpp"
    executable: str = "llama-server"
    endpoint: str | None = None
    threads: int = 4
    context: int = 2048
    gpu_layers: int = 0
    batch_size: int = 256
    ubatch_size: int = 64
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attention: str = "off"
    mmap: bool = True
    cache_prompt: bool = False
    constrain_json: bool = False
    draft_model: str | None = None
    api_key_env: str | None = None
    accelerator: str = "auto"
    device: str | None = None
    draft_max: int | None = None
    supervisor_executable: str | None = None

    def __post_init__(self):
        if self.supervisor_executable is not None and (
            not isinstance(self.supervisor_executable, str)
            or not self.supervisor_executable.strip()
        ):
            raise ValueError("supervisor_executable must identify a native supervisor binary")
        if not self.name or not self.model:
            raise ValueError("Candidate name and model must be nonempty")
        if self.backend not in ("llama.cpp", "openai"):
            raise ValueError("backend must be llama.cpp or openai")
        for key in ("threads", "context", "batch_size", "ubatch_size"):
            positive(getattr(self, key), key, integer=True)
        if type(self.gpu_layers) is not int or self.gpu_layers < -1:
            raise ValueError("gpu_layers must be -1 (all), 0 (CPU), or a positive integer")
        if self.ubatch_size > self.batch_size:
            raise ValueError("ubatch_size cannot exceed batch_size")
        if self.flash_attention not in ("on", "off", "auto"):
            raise ValueError("flash_attention must be on, off or auto")
        for key in ("cache_type_k", "cache_type_v"):
            if getattr(self, key) not in ("f16", "q8_0", "q4_0"):
                raise ValueError(f"unsupported {key}; choose f16, q8_0 or q4_0")
        for key in ("mmap", "cache_prompt", "constrain_json"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        if self.cache_type_v != "f16" and self.flash_attention != "on":
            raise ValueError("quantised V cache requires flash_attention=on")
        if self.backend == "openai" and not self.endpoint:
            raise ValueError("openai backend requires an explicit endpoint")
        if self.accelerator not in ("auto", "cpu", "cuda", "metal", "vulkan", "rocm"):
            raise ValueError("unsupported accelerator")
        if self.accelerator == "cpu" and self.gpu_layers != 0:
            raise ValueError("CPU accelerator requires gpu_layers=0")
        if self.accelerator not in ("auto", "cpu") and self.gpu_layers == 0:
            raise ValueError("an explicit accelerator requires GPU layer offload")
        if self.device is not None and (
            not isinstance(self.device, str)
            or not self.device
            or any(c.isspace() for c in self.device)
            or self.device.startswith("-")
        ):
            raise ValueError("device must be a runtime device identifier")
        if self.draft_max is not None:
            positive(self.draft_max, "draft_max", integer=True)
            if self.draft_max > 64 or not self.draft_model:
                raise ValueError("draft_max requires a draft model and must be <=64")
        if self.backend != "llama.cpp" and (
            self.accelerator != "auto" or self.device is not None or self.draft_max is not None
        ):
            raise ValueError("accelerator and draft settings require a managed llama.cpp backend")


@dataclass(frozen=True)
class Limits:
    min_quality: float = 0.9
    max_rss_gib: float | None = None
    max_gpu_gib: float | None = None
    max_latency_s: float | None = None
    max_ttft_s: float | None = None
    min_available_gib: float = 1.0

    def __post_init__(self):
        finite(self.min_quality, "min_quality")
        if not 0 <= self.min_quality <= 1:
            raise ValueError("min_quality must be in [0, 1]")
        for key in ("max_rss_gib", "max_gpu_gib", "max_latency_s", "max_ttft_s"):
            if getattr(self, key) is not None:
                positive(getattr(self, key), key)
        finite(self.min_available_gib, "min_available_gib")
        if self.min_available_gib < 0:
            raise ValueError("min_available_gib must be nonnegative")


def finite(value: Any, label: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")


def positive(value: Any, label: str, integer=False):
    finite(value, label)
    if value <= 0 or (integer and type(value) is not int):
        raise ValueError(f"{label} must be a positive {'integer' if integer else 'number'}")


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset: str
    candidates: tuple[Candidate, ...]
    repeats: int = 3
    warmup: int = 1
    max_tokens: int = 64
    timeout_s: float = 120
    startup_timeout_s: float = 120
    max_trials: int = 32
    seed: int = 42
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self):
        for key in ("repeats", "max_tokens", "max_trials"):
            positive(getattr(self, key), key, integer=True)
        for key in ("timeout_s", "startup_timeout_s"):
            positive(getattr(self, key), key)
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.warmup) is not int or self.warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        if not self.name or not self.candidates:
            raise ValueError("name and candidates are required")
        if len(self.candidates) > self.max_trials:
            raise ValueError(
                "candidate count exceeds max_trials; narrow the sweep or raise the bound"
            )
        names = [c.name for c in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")
        if any(c.context <= self.max_tokens for c in self.candidates):
            raise ValueError("context must exceed max_tokens to leave room for the prompt")


def _strict(cls, data):
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def parse_experiment(data: dict, base_dir: str | Path = ".") -> Experiment:
    """Load the file schema from an object without creating a temporary config file."""
    base_dir = Path(base_dir).resolve()
    if not isinstance(data, dict):
        raise ValueError("experiment must be a JSON object")
    data = json.loads(json.dumps(data, allow_nan=False))
    candidates = []
    bound = data.get("max_trials", 32)
    positive(bound, "max_trials", integer=True)
    for entry in data.pop("candidates", []):
        base = dict(entry)
        sweep = base.pop("sweep", {})
        if not isinstance(sweep, dict) or any(
            not isinstance(v, list) or not v for v in sweep.values()
        ):
            raise ValueError("sweep must map candidate options to nonempty lists")
        if set(sweep) & {"name", "backend", "endpoint", "executable", "api_key_env"}:
            raise ValueError("sweep only model paths and inference settings")
        if math.prod(len(v) for v in sweep.values()) + len(candidates) > bound:
            raise ValueError("expanded sweep exceeds max_trials")
        for values in itertools.product(*sweep.values()):
            item = {**base, **dict(zip(sweep, values, strict=False))}
            if sweep:
                item["name"] = (
                    base["name"]
                    + "-"
                    + "-".join(f"{k}={v}" for k, v in zip(sweep, values, strict=False))
                )
            if item.get("backend", "llama.cpp") == "llama.cpp":
                for key in ("model", "draft_model"):
                    if item.get(key):
                        item[key] = str((base_dir / item[key]).resolve())
                exe = item.get("executable", "llama-server")
                if "/" in exe or "\\" in exe:
                    item["executable"] = str((base_dir / exe).resolve())
                native = item.get("supervisor_executable")
                if native and ("/" in native or "\\" in native):
                    item["supervisor_executable"] = str((base_dir / native).resolve())
            candidates.append(_strict(Candidate, item))
    data["candidates"] = tuple(candidates)
    data["limits"] = _strict(Limits, data.get("limits", {}))
    data["dataset"] = str((base_dir / data["dataset"]).resolve())
    return _strict(Experiment, data)


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path).resolve()
    return parse_experiment(json.loads(path.read_text(encoding="utf-8")), path.parent)


def digest(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
