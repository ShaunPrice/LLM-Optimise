import json
import threading
import time

import pytest

from llm_optimise import server
from llm_optimise.server import App


def wait_job(app, result):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = app.jobs[result["id"]]
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    pytest.fail("workbench job did not finish")


def test_shared_dataset_dispatch_and_registry(tmp_path):
    app = App(tmp_path)
    rows = [
        {"id": str(i), "prompt": f"item {i}", "expected": "yes", "group": str(i)} for i in range(8)
    ]
    result = app.mutate("/api/workbench/dataset-prepare", {"rows": rows, "name": "test"})
    assert result["unique_rows"] == 8
    assert app.state()["workbench"]["datasets"][0]["name"] == "test"
    evaluated = app.mutate(
        "/api/workbench/dataset-evaluate",
        {"rows": rows, "predictions": {row["id"]: "yes" for row in rows}},
    )
    assert evaluated["quality"] == 1
    failed = dict(evaluated, quality=0)
    compared = app.mutate(
        "/api/workbench/dataset-compare", {"baseline": evaluated, "candidate": failed}
    )
    assert not compared["passed"]


def test_async_workbench_persists_result_and_rejects_heavy_overlap(tmp_path, monkeypatch):
    app = App(tmp_path)
    started, release = threading.Event(), threading.Event()

    def execute(*args, **kwargs):
        started.set()
        release.wait(3)
        return {"status": "passed", "proof": True}

    monkeypatch.setattr(server, "execute_workbench", execute)
    first = app.mutate("/api/workbench/training-probe", {"python": "fixture"})
    assert started.wait(1)
    with pytest.raises(ValueError, match="running"):
        app.mutate("/api/workbench/training-probe", {"python": "fixture"})
    release.set()
    job = wait_job(app, first)
    assert job["result"]["proof"]
    deadline = time.monotonic() + 2
    path = tmp_path / "runs/jobs" / f"{first['id']}.json"
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert json.loads(path.read_text())["result"]["proof"]


def test_job_retention_never_discards_active_cancellation(tmp_path):
    app = App(tmp_path)
    active = {"id": "active", "kind": "chat", "status": "running", "cancel": threading.Event()}
    app.jobs["active"] = active
    for i in range(39):
        app.jobs[str(i)] = {"id": str(i), "status": "complete"}
    wait_job(app, app.start_job("chat", lambda job: {"ok": True}))
    assert app.jobs["active"] is active and "0" not in app.jobs
    app.mutate("/api/cancel", {"id": "active"})
    assert active["cancel"].is_set()


def test_focused_generation_blocks_missing_facts_before_provider(tmp_path):
    root = tmp_path / "projects/demo"
    root.mkdir(parents=True)
    (root / "sensor.py").write_text("threshold = 70\n")
    app = App(tmp_path)
    with pytest.raises(ValueError, match="required facts"):
        app.code(
            {
                "project": "demo",
                "context_files": ["sensor.py"],
                "prompt": "threshold",
                "context_selection": {"max_chars": 256, "required_facts": ["threshold = 80"]},
            }
        )
    assert not app.jobs


def test_managed_supervisor_path_uses_workspace_not_service_cwd(tmp_path, monkeypatch):
    from llm_optimise import managed

    app = App(tmp_path)
    (tmp_path / "model.gguf").write_bytes(b"fixture")
    captured = []

    def backend(candidate, *args):
        captured.append(candidate)
        raise RuntimeError("stop before launch")

    monkeypatch.setattr(managed, "Backend", backend)
    monkeypatch.setattr(managed, "sample_headroom", lambda: {"unified_memory": True})
    try:
        with pytest.raises(RuntimeError, match="stop before launch"):
            app.managed.start(
                {
                    "model": "model.gguf",
                    "gpu_layers": 0,
                    "supervisor_executable": "native/llm-supervisor",
                }
            )
        assert captured[0].supervisor_executable == str(tmp_path / "native/llm-supervisor")
    finally:
        app.managed.close()
