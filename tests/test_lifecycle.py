import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_optimise.lifecycle import (
    LeaseCancelledError,
    ModelLifecycleManager,
    ResourceAdmissionError,
)


class FakeBackend:
    process = None
    pid = None

    def __init__(self, *, gate=None, fail=False):
        self.launched = self.closed = 0
        self.gate, self.fail = gate, fail

    def launch(self):
        self.launched += 1

    def ready(self, timeout, check):
        while self.gate is not None and not self.gate.wait(0.01):
            check()
        check()
        if self.fail:
            raise RuntimeError("load failed")

    def close(self):
        self.closed += 1


def manager(**kwargs):
    kwargs.setdefault(
        "headroom",
        lambda: {"ram_available_gib": 8, "gpu_available_gib": 4, "unified_memory": False},
    )
    kwargs.setdefault("start_reaper", False)
    return ModelLifecycleManager(**kwargs)


def test_deduplicates_concurrent_loading_and_keeps_resident():
    gate = threading.Event()
    first_entered = threading.Event()
    backend = FakeBackend(gate=gate)

    def first_factory():
        first_entered.set()
        return backend

    with manager() as models, ThreadPoolExecutor(2) as pool:
        first = pool.submit(models.acquire, "a", first_factory, ram_gib=2, max_concurrency=2)
        assert first_entered.wait(timeout=2)
        second = pool.submit(
            models.acquire,
            "a",
            lambda: pytest.fail("duplicate factory"),
            ram_gib=2,
            max_concurrency=2,
        )
        time.sleep(0.03)
        gate.set()
        leases = [first.result(2), second.result(2)]
        assert backend.launched == 1
        assert models.status()["active_leases"] == 2
        for lease in leases:
            lease.release()
        assert backend.closed == 0
        with models.acquire("a", lambda: pytest.fail("reloaded"), ram_gib=2, max_concurrency=2):
            pass
    assert backend.closed == 1


def test_loading_reservations_prevent_oversubscription():
    gate = threading.Event()
    with (
        manager(headroom=lambda: {"ram_available_gib": 5}) as models,
        ThreadPoolExecutor(1) as pool,
    ):
        first = pool.submit(models.acquire, "a", lambda: FakeBackend(gate=gate), ram_gib=3)
        for _ in range(100):
            if models.status()["models"]:
                break
            time.sleep(0.005)
        assert models.status()["reserved_ram_gib"] == 3
        with pytest.raises(ResourceAdmissionError, match="RAM"):
            models.acquire("b", FakeBackend, ram_gib=3, timeout_s=0)
        gate.set()
        first.result(2).release()


def test_idle_model_evicted_for_capacity_and_unloads_after_timeout():
    now = [0.0]
    first, second = FakeBackend(), FakeBackend()
    with manager(max_models=1, idle_timeout_s=5, clock=lambda: now[0]) as models:
        models.acquire("a", lambda: first, ram_gib=1).release()
        models.acquire("b", lambda: second, ram_gib=1).release()
        assert first.closed == 1
        now[0] = 6
        assert models.reap_idle() == ["b"]
        assert second.closed == 1


def test_concurrency_limits_and_release_idempotence():
    with manager(max_active_leases=1) as models:
        lease = models.acquire("a", FakeBackend, ram_gib=1)
        with pytest.raises(ResourceAdmissionError, match="lease limit"):
            models.acquire("b", FakeBackend, ram_gib=1, timeout_s=0)
        with pytest.raises(ResourceAdmissionError, match="active leases"):
            models.unload("a")
        lease.release()
        lease.release()
        assert models.status()["active_leases"] == 0


def test_waiting_for_busy_model_does_not_evict_unrelated_idle_model():
    idle = FakeBackend()
    with manager(max_active_leases=3) as models:
        models.acquire("idle", lambda: idle, ram_gib=1).release()
        active = models.acquire("busy", FakeBackend, ram_gib=1)
        with pytest.raises(ResourceAdmissionError, match="concurrency"):
            models.acquire("busy", FakeBackend, ram_gib=1, timeout_s=0)
        assert idle.closed == 0
        active.release()


def test_failed_load_cleans_reservations_and_can_retry():
    broken = FakeBackend(fail=True)
    with manager() as models:
        with pytest.raises(RuntimeError, match="load failed"):
            models.acquire("a", lambda: broken, ram_gib=2)
        assert broken.closed == 1
        assert models.status()["reserved_ram_gib"] == 0
        assert models.status()["models"] == []
        models.acquire("a", FakeBackend, ram_gib=2).release()


def test_cancel_during_loading_and_existing_lease():
    cancel = threading.Event()
    backend = FakeBackend(gate=threading.Event())
    with manager() as models, ThreadPoolExecutor(1) as pool:
        future = pool.submit(models.acquire, "a", lambda: backend, ram_gib=1, cancel=cancel)
        cancel.set()
        with pytest.raises(LeaseCancelledError):
            future.result(2)
        lease = models.acquire("b", FakeBackend, ram_gib=1)
        models.cancel("b")
        with pytest.raises(LeaseCancelledError):
            lease.check()
        lease.release()


def test_discrete_gpu_unknown_fails_closed_and_unified_counts_once():
    with manager(headroom=lambda: {"ram_available_gib": 8}) as models:
        with pytest.raises(ResourceAdmissionError, match="unavailable"):
            models.acquire("a", FakeBackend, ram_gib=1, gpu_gib=1)
    with manager(headroom=lambda: {"ram_available_gib": 8, "unified_memory": True}) as models:
        models.acquire("a", FakeBackend, ram_gib=5, gpu_gib=5).release()


def test_working_set_reservations_apply_per_lease():
    with manager(headroom=lambda: {"ram_available_gib": 4}) as models:
        lease = models.acquire("a", FakeBackend, ram_gib=1, working_ram_gib=2, max_concurrency=2)
        with pytest.raises(ResourceAdmissionError, match="RAM"):
            models.acquire(
                "a", FakeBackend, ram_gib=1, working_ram_gib=2, max_concurrency=2, timeout_s=0
            )
        lease.release()


def test_live_rss_violation_closes_owned_backend_and_revokes_lease():
    backend = FakeBackend()
    with manager(resource_sampler=lambda backend, **kw: {"rss_gib": 3, "gpu_gib": None}) as models:
        lease = models.acquire("a", lambda: backend, ram_gib=1, max_rss_gib=2)
        assert models.monitor_once()[0]["reason"].startswith("sampled rss_gib")
        assert backend.closed == 1
        with pytest.raises(LeaseCancelledError):
            lease.check()


def test_live_headroom_violation_cancels_owned_model():
    available = [8]
    with manager(headroom=lambda: {"ram_available_gib": available[0]}) as models:
        lease = models.acquire("a", FakeBackend, ram_gib=1)
        available[0] = 0.5
        assert models.monitor_once()
        with pytest.raises(LeaseCancelledError):
            lease.check()


def test_explicit_gpu_limit_fails_closed_when_process_telemetry_is_unknown():
    with manager(resource_sampler=lambda backend, **kw: {"rss_gib": 1, "gpu_gib": None}) as models:
        lease = models.acquire("a", FakeBackend, ram_gib=1, gpu_gib=1, max_gpu_gib=2)
        assert "cannot be monitored" in models.monitor_once()[0]["reason"]
        with pytest.raises(LeaseCancelledError):
            lease.check()


def test_cancellation_before_context_entry_does_not_leak_a_lease():
    cancel = threading.Event()
    with manager() as models:
        lease = models.acquire("a", FakeBackend, ram_gib=1, cancel=cancel)
        cancel.set()
        with pytest.raises(LeaseCancelledError), lease:
            pytest.fail("entered cancelled lease")
        assert models.status()["active_leases"] == 0


def test_same_identity_requires_consistent_estimates_and_closed_refuses():
    models = manager()
    models.acquire("a", FakeBackend, ram_gib=1).release()
    with pytest.raises(ValueError, match="different resource"):
        models.acquire("a", FakeBackend, ram_gib=2)
    models.close()
    with pytest.raises(LeaseCancelledError, match="closed"):
        models.acquire("b", FakeBackend, ram_gib=1)


@pytest.mark.parametrize(
    "field,value",
    [("ram_gib", -1), ("ram_gib", float("nan")), ("gpu_gib", True), ("max_concurrency", 0)],
)
def test_rejects_invalid_resource_inputs(field, value):
    with manager() as models, pytest.raises(ValueError):
        models.acquire("a", FakeBackend, **{"ram_gib": 1, field: value})
