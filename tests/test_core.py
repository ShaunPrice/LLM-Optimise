import json
import math

import pytest

from llm_optimise.config import Candidate, Limits, load_experiment
from llm_optimise.planner import estimate_memory
from llm_optimise.quality import Task, load_tasks
from llm_optimise.runner import aggregate, eligibility, pareto_frontier
from llm_optimise.training import training_recipe
from llm_optimise.workspace import apply_proposal, prepare_proposal, safe_path


def test_config_paths_sweep_and_budget(tmp_path):
    data = {
        "name": "test",
        "dataset": "tasks.jsonl",
        "candidates": [
            {
                "name": "model",
                "model": "a.gguf",
                "sweep": {"threads": [1, 2], "context": [512, 1024]},
            }
        ],
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(data))
    exp = load_experiment(path)
    assert len(exp.candidates) == 4
    assert exp.dataset == str(tmp_path / "tasks.jsonl")
    assert exp.candidates[0].model == str(tmp_path / "a.gguf")
    data["max_trials"] = 3
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="max_trials"):
        load_experiment(path)


@pytest.mark.parametrize(
    "kw",
    [
        {"context": 0},
        {"threads": True},
        {"gpu_layers": -2},
        {"cache_type_v": "q4_0"},
        {"ubatch_size": 512, "batch_size": 64},
        {"mmap": "false"},
    ],
)
def test_bad_candidates(kw):
    with pytest.raises(ValueError):
        Candidate(name="test", model="m.gguf", **kw)


def test_quality_is_strict_and_does_not_execute():
    assert Task("t", "prompt", "negative").score("The sentiment is negative") == 0
    assert Task("t", "prompt", "negative").score("  Negative\n") == 1
    assert Task("t", "prompt", {"n": 1}, "json").score('{"n":true}') == 0
    assert Task("t", "prompt", {"n": 1}, "json_subset").score('{"n":1,"extra":2}') == 1
    assert Task("t", "prompt", 42, "numeric").score("nan") == 0
    assert Task("t", "prompt", 42, "numeric").score("42") == 1
    assert Task("t", "prompt", 42, "numeric").score("__import__('os')") == 0


def test_no_empty_or_duplicate_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("")
    with pytest.raises(ValueError):
        load_tasks(path)
    line = json.dumps({"id": "a", "prompt": "hello", "expected": "hi"}) + "\n"
    path.write_text(line * 2)
    with pytest.raises(ValueError):
        load_tasks(path)


def trial(name, speed, ram, gpu=None, quality=1):
    return {
        "name": name,
        "status": "complete",
        "eligible": True,
        "metrics": {
            "quality": quality,
            "latency_p50_s": speed,
            "latency_p95_s": speed,
            "ttft_p95_s": 0.1,
        },
        "memory": {"rss_peak_gib": ram, "gpu_peak_gib": gpu},
    }


def test_frontier_respects_quality_missing_telemetry_and_failures():
    a, b, c = trial("a", 1, 1), trial("b", 2, 2), trial("c", 0.5, 1, 0)
    failed = trial("failed", 0.01, 0.01)
    failed["eligible"] = False
    assert pareto_frontier([a, b, c, failed]) == ["a", "c"]
    assert "GPU memory unavailable" in eligibility(a, Limits(max_gpu_gib=2))
    assert eligibility(c, Limits(max_gpu_gib=2)) == []
    assert "quality below minimum" in eligibility(trial("bad", 0.01, 0.1, quality=0.2), Limits())


def test_throughput_aggregation_is_time_weighted():
    rows = [
        {
            "latency_s": 2,
            "output_tokens": 10,
            "decode_tokens_s": 10,
            "ttft_s": 0.1,
            "score": 1,
            "truncated": False,
            "task_id": "a",
        },
        {
            "latency_s": 11,
            "output_tokens": 10,
            "decode_tokens_s": 1,
            "ttft_s": 0.2,
            "score": 0,
            "truncated": True,
            "task_id": "b",
        },
    ]
    value = aggregate(rows)
    assert value["decode_tokens_s"] == pytest.approx(20 / 11)
    assert value["end_to_end_tokens_s"] == pytest.approx(20 / 13)
    assert value["quality"] == 0.5
    rows[0]["output_tokens"] = None
    assert aggregate(rows)["end_to_end_tokens_s"] is None


def test_dense_kv_formula_and_unified_pool():
    result = estimate_memory(1, 4, 20, 2, 64, 1024, unified=True)
    assert result["kv_payload_gib"] * 1024**3 == 2 * 20 * 2 * 64 * 1024 * 2
    assert result["estimated_device_gib"] is None
    with pytest.raises(ValueError):
        estimate_memory(1, 4, 20, 2, 64, 1024, gpu_fraction=math.nan)


@pytest.mark.parametrize(
    "name",
    [
        "../x.py",
        "/tmp/x.py",
        ".env",
        "x/.git/config",
        "C:\\evil.py",
        "x\\..\\evil.py",
        "NUL.txt",
        "file.",
    ],
)
def test_unsafe_code_paths(tmp_path, name):
    with pytest.raises(ValueError):
        safe_path(tmp_path, name)


def test_review_apply_and_stale_detection(tmp_path):
    (tmp_path / "a.py").write_text("old\n")
    response = json.dumps(
        {
            "summary": "Update a",
            "files": [
                {"path": "a.py", "content": "new\n"},
                {"path": "src/b.py", "content": "ok\n"},
            ],
        }
    )
    proposal = prepare_proposal(tmp_path, response)
    assert "-old" in proposal["files"][0]["diff"]
    (tmp_path / "a.py").write_text("concurrent\n")
    with pytest.raises(ValueError, match="changed since preview"):
        apply_proposal(tmp_path, proposal)
    assert not (tmp_path / "src/b.py").exists()
    (tmp_path / "a.py").write_text("old\n")
    assert apply_proposal(tmp_path, proposal) == ["a.py", "src/b.py"]
    with pytest.raises(ValueError, match="already applied"):
        apply_proposal(tmp_path, proposal)


def test_no_case_collision_or_file_parent_collision(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        prepare_proposal(
            tmp_path,
            json.dumps(
                {
                    "summary": "x",
                    "files": [{"path": "A.py", "content": "x"}, {"path": "a.py", "content": "y"}],
                }
            ),
        )
    p = prepare_proposal(
        tmp_path,
        json.dumps(
            {
                "summary": "x",
                "files": [{"path": "a", "content": "x"}, {"path": "a/b", "content": "y"}],
            }
        ),
    )
    with pytest.raises(ValueError, match="collision"):
        apply_proposal(tmp_path, p)
    assert not (tmp_path / "a").exists()


def test_symlinks_rejected(tmp_path):
    other = tmp_path / "outside"
    other.mkdir()
    try:
        (tmp_path / "link").symlink_to(other, target_is_directory=True)
    except OSError:
        pytest.skip("host does not grant symlink creation")
    with pytest.raises(ValueError, match="symlinks"):
        safe_path(tmp_path, "link/a.py")


def test_soup_recipe_uses_data_max_length():
    value = training_recipe("soup-stream", "model", "data.jsonl", "output")
    assert value["config"]["data"]["max_length"] == 512
    assert "max_length" not in value["config"]["training"]
    assert value["config"]["training"]["stream_layers"] is True
    mlx = training_recipe("soup-mlx", "model", "data", "output")["config"]
    assert mlx["backend"] == "mlx" and "stream_layers" not in mlx["training"]


def test_code_context_must_not_change_during_generation(tmp_path):
    from llm_optimise.workspace import prepare_proposal, read_context

    (tmp_path / "main.py").write_text("old")
    context = read_context(tmp_path, ["main.py"])
    (tmp_path / "main.py").write_text("edited while model ran")
    with pytest.raises(ValueError, match="changed during generation"):
        prepare_proposal(tmp_path, '{"summary":"x","files":[]}', context)


def test_failed_atomic_apply_preserves_existing_file(tmp_path, monkeypatch):
    from llm_optimise import workspace

    (tmp_path / "main.py").write_text("original")
    proposal = workspace.prepare_proposal(
        tmp_path, '{"summary":"x","files":[{"path":"main.py","content":"replacement"}]}'
    )

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(workspace.os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        workspace.apply_proposal(tmp_path, proposal)
    assert (tmp_path / "main.py").read_text() == "original"
    assert not list(tmp_path.glob(".llm-optimise-*"))
