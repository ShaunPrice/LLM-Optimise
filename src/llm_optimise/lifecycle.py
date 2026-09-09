"""Thread-safe, memory-aware residency and leases for injected inference backends.

Admission uses live free memory and conservative estimates. Reservations coordinate
this manager's work; they are not OS memory limits or guarantees against other apps.
"""

from __future__ import annotations

import math
import platform
import threading
import time
import uuid
from dataclasses import dataclass, field

import psutil

from .hardware import command_output, process_gpu_bytes


class ResourceAdmissionError(RuntimeError):
    """A model/request cannot be admitted within the configured resource policy."""


class LeaseCancelledError(RuntimeError):
    """The lease, its caller, or its underlying model has been cancelled."""


def sample_headroom():
    """Return measured RAM and conservative minimum discrete GPU headroom in GiB."""
    memory = psutil.virtual_memory()
    unified = platform.system() == "Darwin" and platform.machine() == "arm64"
    gpu_free = None
    if not unified:
        value = command_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=2,
        )
        if value:
            try:
                gpu_free = min(float(line.strip()) / 1024 for line in value.splitlines())
            except ValueError:
                pass
    return {
        "ram_available_gib": memory.available / 1024**3,
        "gpu_available_gib": gpu_free,
        "unified_memory": unified,
        "sampled_at": time.time(),
        "gpu_scope": "minimum free memory across reported NVIDIA GPUs",
    }


def sample_backend_resources(backend, *, include_gpu=False):
    """Sample only an injected backend's owned process tree; absent values stay unknown."""
    pid = getattr(backend, "pid", None)
    if pid is None:
        return {"rss_gib": None, "gpu_gib": None}
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        pids, rss = set(), 0
        for process in processes:
            try:
                rss += process.memory_info().rss
                pids.add(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        gpu = process_gpu_bytes(pids) if include_gpu and pids else None
        return {
            "rss_gib": rss / 1024**3 if pids else None,
            "gpu_gib": None if gpu is None else gpu / 1024**3,
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"rss_gib": None, "gpu_gib": None}


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return float(value)


@dataclass
class _Entry:
    key: str
    backend: object
    requirements: tuple
    state: str = "loading"
    leases: dict = field(default_factory=dict)
    last_used: float = 0
    stop: threading.Event = field(default_factory=threading.Event)
    metrics: dict = field(default_factory=dict)


class Lease:
    """Release with ``with`` or ``release()``; call ``check()`` during generation."""

    def __init__(self, manager, entry, token, caller_cancel):
        self._manager, self._entry, self._token = manager, entry, token
        self._cancel = caller_cancel
        self._released = False

    @property
    def backend(self):
        self.check()
        return self._entry.backend

    @property
    def key(self):
        return self._entry.key

    def check(self):
        if (
            self._released
            or self._entry.stop.is_set()
            or (self._cancel is not None and self._cancel.is_set())
        ):
            raise LeaseCancelledError(f"model lease cancelled: {self.key}")
        process = getattr(self._entry.backend, "process", None)
        if process is not None and process.poll() is not None:
            self._manager.cancel(self.key)
            raise LeaseCancelledError(f"model process exited: {self.key}")

    def release(self):
        if not self._released:
            self._released = True
            self._manager._release(self._entry, self._token)

    def __enter__(self):
        try:
            self.check()
        except BaseException:
            self.release()
            raise
        return self

    def __exit__(self, *_):
        self.release()


class ModelLifecycleManager:
    """Deduplicated residency with bounded leases, admission and idle unloading.

    ``ram_gib`` includes the complete resident footprint on unified-memory hosts;
    GPU residency is not added to it again. On discrete GPUs provide RAM and GPU
    separately. Working memory is reserved per active lease. Loaded models are
    already reflected in measured free memory, so their base estimate is only
    reserved while loading. Estimates must be conservative for the selected
    context/batch/runtime; a warmed measurement is preferable to a weight size.
    """

    def __init__(
        self,
        *,
        headroom=sample_headroom,
        max_active_leases=2,
        max_models=2,
        min_ram_gib=1,
        min_gpu_gib=0.25,
        idle_timeout_s=60,
        start_reaper=True,
        clock=time.monotonic,
        resource_sampler=sample_backend_resources,
        monitor_interval_s=0.5,
    ):
        for name, value in (("max_active_leases", max_active_leases), ("max_models", max_models)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_active_leases, self.max_models = max_active_leases, max_models
        self.min_ram_gib = _number(min_ram_gib, "min_ram_gib")
        self.min_gpu_gib = _number(min_gpu_gib, "min_gpu_gib")
        self.idle_timeout_s = _number(idle_timeout_s, "idle_timeout_s")
        self._headroom, self._clock = headroom, clock
        self._resource_sampler = resource_sampler
        self.monitor_interval_s = _number(monitor_interval_s, "monitor_interval_s", positive=True)
        self._condition = threading.Condition(threading.RLock())
        self._entries = {}
        self._closed = False
        self._last_headroom = None
        self._last_error = None
        self._stop = threading.Event()
        self._thread = None
        if start_reaper:
            self._thread = threading.Thread(target=self._reaper, daemon=True, name="llm-residency")
            self._thread.start()

    def _reservations(self):
        ram = gpu = 0.0
        for entry in self._entries.values():
            base_ram, base_gpu, work_ram, work_gpu, *_ = entry.requirements
            if entry.state == "loading":
                ram += base_ram
                gpu += base_gpu
            ram += len(entry.leases) * work_ram
            gpu += len(entry.leases) * work_gpu
        return ram, gpu

    def _admission(self, ram, gpu):
        snapshot = dict(self._headroom())
        self._last_headroom = snapshot
        free_ram = _number(snapshot.get("ram_available_gib"), "measured ram_available_gib")
        reserved_ram, reserved_gpu = self._reservations()
        if free_ram - reserved_ram - self.min_ram_gib < ram:
            return "insufficient measured available RAM after reservations and headroom"
        if not snapshot.get("unified_memory", False) and gpu > 0:
            free_gpu = snapshot.get("gpu_available_gib")
            if free_gpu is None:
                raise ResourceAdmissionError(
                    "discrete GPU headroom is unavailable; admission refused"
                )
            free_gpu = _number(free_gpu, "measured gpu_available_gib")
            if free_gpu - reserved_gpu - self.min_gpu_gib < gpu:
                return "insufficient measured GPU headroom after reservations"
        return None

    def acquire(
        self,
        key,
        factory,
        *,
        ram_gib,
        gpu_gib=0,
        working_ram_gib=0,
        working_gpu_gib=0,
        max_concurrency=1,
        timeout_s=60,
        cancel=None,
        ready_timeout_s=60,
        max_rss_gib=None,
        max_gpu_gib=None,
    ):
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a nonempty model/configuration identity")
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        requirements = tuple(
            _number(value, name)
            for value, name in (
                (ram_gib, "ram_gib"),
                (gpu_gib, "gpu_gib"),
                (working_ram_gib, "working_ram_gib"),
                (working_gpu_gib, "working_gpu_gib"),
            )
        ) + (
            max_concurrency,
            None if max_rss_gib is None else _number(max_rss_gib, "max_rss_gib", positive=True),
            None if max_gpu_gib is None else _number(max_gpu_gib, "max_gpu_gib", positive=True),
        )
        timeout_s = _number(timeout_s, "timeout_s")
        ready_timeout_s = _number(ready_timeout_s, "ready_timeout_s", positive=True)
        deadline = self._clock() + timeout_s
        token = uuid.uuid4().hex
        entry = None
        while entry is None:
            victim = None
            with self._condition:
                if self._closed:
                    raise LeaseCancelledError("model manager is closed")
                if cancel is not None and cancel.is_set():
                    raise LeaseCancelledError("model acquisition cancelled")
                existing = self._entries.get(key)
                reason = None
                active = sum(len(item.leases) for item in self._entries.values())
                if existing and existing.requirements != requirements:
                    raise ValueError(
                        "same model key was requested with different resource/concurrency estimates"
                    )
                if active >= self.max_active_leases:
                    reason = "global active lease limit reached"
                elif existing and existing.state != "ready":
                    reason = "model is already loading"
                elif existing and len(existing.leases) >= max_concurrency:
                    reason = "model concurrency limit reached"
                elif existing:
                    reason = self._admission(requirements[2], requirements[3])
                    if reason is None:
                        existing.leases[token] = self._clock()
                        lease = Lease(self, existing, token, cancel)
                        try:
                            lease.check()
                        except BaseException:
                            lease.release()
                            raise
                        return lease
                else:
                    reason = (
                        "resident model limit reached"
                        if len(self._entries) >= self.max_models
                        else self._admission(
                            requirements[0] + requirements[2], requirements[1] + requirements[3]
                        )
                    )
                    if reason is None:
                        # Factory must only construct; process work belongs in launch().
                        entry = _Entry(key, factory(), requirements, last_used=self._clock())
                        entry.leases[token] = self._clock()
                        self._entries[key] = entry
                        break
                # Reclaim only idle resident models, never active work or a load.
                idle = [
                    item
                    for item in self._entries.values()
                    if not item.leases and item.state == "ready" and item.key != key
                ]
                evictable_reason = reason == "resident model limit reached" or (
                    reason is not None and reason.startswith("insufficient measured")
                )
                if idle and evictable_reason:
                    victim = min(idle, key=lambda item: item.last_used)
                    self._entries.pop(victim.key)
                    victim.stop.set()
                elif self._clock() >= deadline:
                    self._last_error = reason
                    raise ResourceAdmissionError(reason or "model admission timed out")
                else:
                    self._condition.wait(timeout=min(0.05, max(0, deadline - self._clock())))
            if victim:
                self._close_backend(victim)
        lease = Lease(self, entry, token, cancel)
        try:
            lease.check()
            entry.backend.launch()
            entry.backend.ready(
                min(ready_timeout_s, max(0.001, deadline - self._clock())), lease.check
            )
            lease.check()
            with self._condition:
                if self._closed or self._entries.get(key) is not entry:
                    raise LeaseCancelledError("model load was cancelled")
                entry.state = "ready"
                self._condition.notify_all()
            return lease
        except BaseException as exc:
            with self._condition:
                if self._entries.get(key) is entry:
                    self._entries.pop(key)
                entry.stop.set()
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._condition.notify_all()
            self._close_backend(entry)
            raise

    def _release(self, entry, token):
        with self._condition:
            entry.leases.pop(token, None)
            entry.last_used = self._clock()
            self._condition.notify_all()

    def _close_backend(self, entry):
        try:
            entry.backend.close()
        except Exception as exc:
            with self._condition:
                self._last_error = f"cleanup failed for {entry.key}: {type(exc).__name__}"

    def unload(self, key, *, force=False):
        with self._condition:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.leases and not force:
                raise ResourceAdmissionError("model has active leases")
            self._entries.pop(key)
            entry.stop.set()
            self._condition.notify_all()
        self._close_backend(entry)
        return True

    def cancel(self, key):
        """Revoke all leases for this model and close its owned backend."""
        return self.unload(key, force=True)

    def reap_idle(self):
        with self._condition:
            keys = [
                item.key
                for item in self._entries.values()
                if not item.leases
                and item.state == "ready"
                and self._clock() - item.last_used >= self.idle_timeout_s
            ]
            entries = [self._entries.pop(key) for key in keys]
            for entry in entries:
                entry.stop.set()
            self._condition.notify_all()
        for entry in entries:
            self._close_backend(entry)
        return keys

    def _reaper(self):
        while not self._stop.wait(self.monitor_interval_s):
            self.monitor_once()
            self.reap_idle()

    def monitor_once(self):
        """Cancel and close owned models on sampled pressure/limits; not an OS hard cap."""
        with self._condition:
            entries = list(self._entries.values())
        if not entries:
            return []
        violations = []
        try:
            snapshot = dict(self._headroom())
            with self._condition:
                self._last_headroom = snapshot
            pressure = snapshot.get("ram_available_gib", float("inf")) < self.min_ram_gib
            for entry in entries:
                if entry.stop.is_set():
                    continue
                process = getattr(entry.backend, "process", None)
                reason = (
                    "owned model process exited"
                    if process is not None and process.poll() is not None
                    else None
                )
                if pressure:
                    reason = "measured available RAM fell below configured headroom"
                metrics = self._resource_sampler(
                    entry.backend, include_gpu=entry.requirements[6] is not None
                )
                with self._condition:
                    entry.metrics = metrics
                for name, limit in (
                    ("rss_gib", entry.requirements[5]),
                    ("gpu_gib", entry.requirements[6]),
                ):
                    if limit is not None and metrics.get(name) is None and entry.state == "ready":
                        reason = f"configured {name} limit cannot be monitored on this backend"
                    if (
                        limit is not None
                        and metrics.get(name) is not None
                        and metrics[name] > limit
                    ):
                        reason = f"sampled {name} exceeded configured model limit"
                if reason:
                    # Identity check prevents a stale sample cancelling a replacement model.
                    with self._condition:
                        if self._entries.get(entry.key) is not entry:
                            continue
                        self._entries.pop(entry.key)
                        entry.stop.set()
                        self._last_error = reason
                        self._condition.notify_all()
                    self._close_backend(entry)
                    violations.append({"key": entry.key, "reason": reason})
        except Exception as exc:
            with self._condition:
                self._last_error = f"resource sampler unavailable: {type(exc).__name__}"
        return violations

    def status(self):
        with self._condition:
            ram, gpu = self._reservations()
            return {
                "closed": self._closed,
                "active_leases": sum(len(item.leases) for item in self._entries.values()),
                "max_active_leases": self.max_active_leases,
                "max_models": self.max_models,
                "reserved_ram_gib": ram,
                "reserved_gpu_gib": gpu,
                "last_headroom": self._last_headroom,
                "last_error": self._last_error,
                "monitor_interval_s": self.monitor_interval_s,
                "measurement_scope": "sampled owned process tree RSS/NVIDIA memory; short peaks may be missed; unknown GPU memory is null",
                "models": [
                    {
                        "key": item.key,
                        "state": item.state,
                        "leases": len(item.leases),
                        "pid": getattr(item.backend, "pid", None),
                        "idle_seconds": 0
                        if item.leases
                        else max(0, self._clock() - item.last_used),
                        "metrics": item.metrics,
                    }
                    for item in self._entries.values()
                ],
            }

    def close(self):
        self._stop.set()
        with self._condition:
            self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()
            for entry in entries:
                entry.stop.set()
            self._condition.notify_all()
        for entry in entries:
            self._close_backend(entry)
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
