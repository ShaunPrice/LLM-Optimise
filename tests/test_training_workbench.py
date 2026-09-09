import json
import sys
import threading
from pathlib import Path

import pytest

from llm_optimise import datasets, training_jobs
from llm_optimise.datasets import (
    compare_regression,
    evaluate_predictions,
    normalise_rows,
    prepare_dataset,
    validate_json_schema,
    validate_schema_definition,
)
from llm_optimise.distillation import distill
from llm_optimise.quality import load_tasks


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "anything"},
        {"type": ["string", "null"]},
        {"pattern": ".*"},
        {"format": "email"},
        {"anyOf": []},
        {"additionalProperties": {}},
        {"required": ["a", "a"]},
        {"minItems": -1},
        {"minimum": float("nan")},
        {"minLength": 3, "maxLength": 2},
        {"enum": []},
        {"items": False},
    ],
)
def test_unsupported_or_invalid_schema_fails_closed(schema):
    with pytest.raises(ValueError):
        validate_schema_definition(schema)


@pytest.mark.parametrize(
    ("value", "schema", "valid"),
    [
        (True, {"type": "integer"}, False),
        (1, {"enum": [True]}, False),
        (
            {"count": 2},
            {
                "type": "object",
                "required": ["count"],
                "properties": {"count": {"type": "integer"}},
                "additionalProperties": False,
            },
            True,
        ),
        (
            {"count": 2, "extra": 3},
            {"properties": {"count": {}}, "additionalProperties": False},
            False,
        ),
        ({}, {"required": ["x"]}, False),
        ([1, "2"], {"type": "array", "items": {"type": "number"}}, False),
        ([1, 2], {"minItems": 2, "maxItems": 2}, True),
        ("long", {"maxLength": 3}, False),
        (float("nan"), {}, False),
        ({"x": float("inf")}, {}, False),
        (1.5, {"type": "integer"}, False),
        (2.0, {"type": "integer", "minimum": 1, "maximum": 2}, True),
        (None, {"type": "null"}, True),
    ],
)
def test_schema_values(value, schema, valid):
    assert validate_json_schema(value, schema)["valid"] is valid


def rows(count=60):
    return [
        {
            "id": f"t{i}",
            "prompt": f"Question {i}",
            "expected": "yes",
            "group": f"document-{i // 3}",
            "task_class": "classification",
        }
        for i in range(count)
    ]


def test_split_is_group_disjoint_stable_and_load_tasks_compatible(tmp_path):
    before = prepare_dataset({"rows": rows(), "seed": 17}, tmp_path / "first")
    after = prepare_dataset({"rows": list(reversed(rows())), "seed": 17}, tmp_path / "second")

    def assignments(manifest):
        return {row["id"]: row["split"] for row in manifest["rows"]}

    assert assignments(before) == assignments(after)
    assert all(
        len({r["split"] for r in before["rows"] if r["group"] == group}) == 1
        for group in {r["group"] for r in before["rows"]}
    )
    for split in before["splits"].values():
        if split["rows"]:
            assert len(load_tasks(split["tasks"])) == split["rows"]
            assert "messages" in json.loads(Path(split["supervised"]).read_text().splitlines()[0])
    extended = prepare_dataset(
        {
            "rows": rows()
            + [{"id": "new", "prompt": "New independent group", "expected": "yes", "group": "new"}],
            "seed": 17,
        },
        tmp_path / "extended",
    )
    assert all(assignments(extended)[key] == value for key, value in assignments(before).items())


def test_cross_group_duplicate_links_whole_groups_without_leakage(tmp_path):
    source = [
        {"id": "a", "prompt": "duplicate", "expected": "yes", "group": "g1"},
        {"id": "b", "prompt": "duplicate", "expected": "yes", "group": "g2"},
        {"id": "c", "prompt": "different", "expected": "yes", "group": "g2"},
    ]
    result = prepare_dataset({"rows": source}, tmp_path)
    assert result["unique_rows"] == 2
    assert len({r["group"] for r in result["rows"]}) == 1
    assert len({r["split"] for r in result["rows"]}) == 1


def test_duplicate_unlabelled_ids_are_deduplicated(tmp_path):
    source = [{"prompt": "same", "expected": "yes"}] * 2
    assert prepare_dataset({"rows": source}, tmp_path)["unique_rows"] == 1


def test_conflicting_duplicates_rejected(tmp_path):
    source = [
        {"id": "a", "prompt": "same", "expected": "yes"},
        {"id": "b", "prompt": "same", "expected": "no"},
    ]
    with pytest.raises(ValueError, match="conflicting"):
        prepare_dataset({"rows": source}, tmp_path)


def test_dataset_rejects_overwrite_and_unsupported_fields(tmp_path):
    prepare_dataset({"rows": rows()}, tmp_path)
    with pytest.raises(ValueError, match="empty"):
        prepare_dataset({"rows": rows()}, tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        normalise_rows([{**rows(1)[0], "command": "never execute"}])


def test_dataset_alpaca_import(tmp_path):
    result = prepare_dataset(
        {"rows": [{"instruction": "Classify", "input": "hello", "output": "greeting"}]}, tmp_path
    )
    assert result["rows"][0]["prompt"] == "Classify\n\nhello"
    assert result["rows"][0]["expected"] == "greeting"


def test_dataset_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "MAX_DATASET_BYTES", 16)
    with pytest.raises(ValueError):
        prepare_dataset({"rows": rows()}, tmp_path)


def test_prediction_missing_schema_and_error_costs():
    source = [
        {
            "id": "a",
            "prompt": "JSON",
            "expected": {"n": 1},
            "evaluator": "json",
            "error_cost": 5,
            "schema_error_cost": 2,
            "json_schema": {
                "type": "object",
                "required": ["n"],
                "properties": {"n": {"type": "integer"}},
            },
        }
    ]
    good = evaluate_predictions(source, {"a": '{"n":1}'})
    bad = evaluate_predictions(source, {"a": '{"n":true}'})
    missing = evaluate_predictions(source, {})
    assert good["quality"] == 1
    assert bad["schema_failures"] == 1 and bad["total_error_cost"] == 7
    assert missing["missing_outputs"] == 1 and missing["quality"] == 0
    assert compare_regression(good, bad)["passed"] is False
    with pytest.raises(ValueError, match="fingerprint"):
        compare_regression(good, {**bad, "dataset_sha256": "different"})


def test_regressions_cannot_be_hidden_by_equal_aggregate_score():
    source = rows(2)
    baseline = evaluate_predictions(source, {"t0": "yes", "t1": "no"})
    candidate = evaluate_predictions(source, {"t0": "no", "t1": "yes"})
    assert baseline["quality"] == candidate["quality"]
    result = compare_regression(baseline, candidate)
    assert not result["passed"] and result["regressed_task_ids"] == ["t0"]
    assert compare_regression(baseline, candidate, {"max_regressed_tasks": 1})["passed"]


def test_prediction_unknown_duplicate_and_nontext_rejected():
    for prediction in ({"other": "yes"}, {"t0": 1}, [{"id": "t0", "output": "yes"}] * 2):
        with pytest.raises(ValueError):
            evaluate_predictions(rows(1), prediction)


def test_command_timeout_terminates_real_worker(tmp_path):
    result = training_jobs.run_command(
        [sys.executable, "-c", "import time;time.sleep(30)"],
        tmp_path,
        timeout_s=0.15,
        min_available_gib=0,
    )
    assert result["status"] == "timeout"
    assert result["elapsed_s"] < 8
    assert result["exit_code"] is not None


def test_command_cancel_before_launch_does_not_execute(tmp_path):
    event = threading.Event()
    event.set()
    result = training_jobs.run_command(["definitely-not-an-executable"], tmp_path, cancel=event)
    assert result["status"] == "cancelled" and result["exit_code"] is None


def test_command_live_cancel_and_progress(tmp_path):
    event = threading.Event()
    timer = threading.Timer(0.7, event.set)
    timer.start()
    updates = []
    try:
        result = training_jobs.run_command(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            tmp_path,
            timeout_s=10,
            min_available_gib=0,
            cancel=event,
            progress=updates.append,
        )
    finally:
        timer.cancel()
    assert result["status"] == "cancelled"
    assert any(x["stage"] == "running" for x in updates)


def test_command_real_rss_limit(tmp_path):
    result = training_jobs.run_command(
        [sys.executable, "-c", "import time;data=bytearray(64*1024*1024);time.sleep(30)"],
        tmp_path,
        timeout_s=10,
        max_rss_gib=0.02,
        min_available_gib=0,
    )
    assert result["status"] == "resource_limit"
    assert "RSS" in result["reason"]


def test_command_output_and_exit_status(tmp_path):
    result = training_jobs.run_command(
        [sys.executable, "-c", "print('actual stdout');raise SystemExit(3)"],
        tmp_path,
        min_available_gib=0,
    )
    assert result["status"] == "failed" and result["exit_code"] == 3
    assert "actual stdout" in result["log_tail"]


def test_unreviewed_soup_environment_fails_closed(monkeypatch):
    class Reply:
        returncode = 0
        stdout = '{"soup_source_sha256":"wrong"}'

    monkeypatch.setattr(training_jobs.subprocess, "run", lambda *a, **k: Reply())
    with pytest.raises(ValueError, match="reviewed"):
        training_jobs.probe_environment(sys.executable)


def adapter_fixture(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    (source / "adapters.safetensors").write_bytes(b"fixture weights, never loaded")
    (source / "adapter_config.json").write_text('{"fine_tune_type":"lora"}')
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base fixture")
    return {
        "path": str(source),
        "base_model": str(base),
        "backend": "mlx",
        "python": sys.executable,
    }


def test_registration_snapshots_weights_and_detects_mutation(tmp_path):
    request = adapter_fixture(tmp_path)
    registry = tmp_path / "registry"
    manifest = training_jobs.register_adapter(request, registry)
    assert training_jobs.list_adapters(registry)[0]["reload_status"] == "unverified"
    Path(request["path"], "adapters.safetensors").write_bytes(b"changed original")
    assert training_jobs._adapter(manifest["id"], registry)["id"] == manifest["id"]
    Path(manifest["path"], "adapters.safetensors").write_bytes(b"changed snapshot")
    with pytest.raises(ValueError, match="modified"):
        training_jobs._adapter(manifest["id"], registry)


def test_registration_base_revision_guard_and_safe_id(tmp_path):
    request = adapter_fixture(tmp_path)
    registry = tmp_path / "registry"
    manifest = training_jobs.register_adapter(request, registry)
    Path(request["base_model"], "model.safetensors").write_bytes(b"changed base")
    with pytest.raises(ValueError, match="base model changed"):
        training_jobs._adapter(manifest["id"], registry)
    with pytest.raises(ValueError, match="invalid adapter ID"):
        training_jobs._adapter("../escape", registry)


def test_adapter_provenance_is_a_snapshot_not_a_circular_reference(tmp_path):
    request = adapter_fixture(tmp_path)
    result = {"status": "passed", "test": "fixture"}
    manifest = training_jobs.register_adapter(
        {**request, "training_result": result}, tmp_path / "registry"
    )
    result["adapter"] = manifest
    json.dumps(result)
    assert "adapter" not in manifest["training_result"]


def test_training_budget_and_remote_model_refused_before_launch(tmp_path):
    model = tmp_path / "base"
    model.mkdir()
    (model / "weights").write_bytes(b"fixture")
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps({"instruction": "q", "output": "a"}) for _ in range(8)))
    request = {
        "config": {
            "base": str(model),
            "task": "sft",
            "backend": "mlx",
            "data": {"train": str(data), "format": "alpaca"},
            "training": {"epochs": 1, "batch_size": 1},
        },
        "max_steps": 2,
    }
    with pytest.raises(ValueError, match="max_steps"):
        training_jobs.run_training(request, tmp_path / "job1")
    request["config"]["base"] = "private/model"
    with pytest.raises(ValueError, match="downloaded local"):
        training_jobs.run_training(request, tmp_path / "job2")


def teacher(response="yes", cost=0.001, calls=None):
    def callback(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return {
            "route": {"model_id": "configured-teacher"},
            "completion": {
                "text": response,
                "accounted_cost_usd": cost,
                "input_tokens": 10,
                "output_tokens": 2,
            },
        }

    return callback


def distill_request(**updates):
    return {
        "rows": rows(3),
        "data_authorized": True,
        "max_requests": 2,
        "max_cost_usd": 0.005,
        **updates,
    }


def test_distillation_requires_authorization_before_callback(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="data_authorized"):
        distill(distill_request(data_authorized=False), tmp_path, teacher(calls=calls))
    assert calls == []


def test_distillation_honours_request_and_remaining_cost_budgets(tmp_path):
    calls = []
    result = distill(distill_request(), tmp_path, teacher(calls=calls))
    assert result["status"] == "budget_exhausted"
    assert result["accepted_rows"] == 2 and result["requests_attempted"] == 2
    assert [x["max_cost_usd"] for x in calls] == pytest.approx([0.005, 0.004])
    assert len(load_tasks(result["tasks_path"])) == 2
    assert result["accounted_cost_usd"] == pytest.approx(0.002)


def test_distillation_rejects_wrong_answers_and_keeps_audit(tmp_path):
    result = distill(distill_request(), tmp_path, teacher(response="no"))
    assert result["accepted_rows"] == 0 and result["rejected_rows"] == 2
    assert all(x["reference_score"] == 0 for x in result["audit"])


def test_distillation_schema_only_records_limited_evidence(tmp_path):
    request = distill_request(
        rows=[
            {
                "id": "t",
                "prompt": "Return JSON",
                "json_schema": {
                    "type": "object",
                    "required": ["ok"],
                    "properties": {"ok": {"type": "boolean"}},
                },
            }
        ]
    )
    result = distill(request, tmp_path, teacher(response='{"ok":true}'))
    assert result["accepted_rows"] == 1
    assert result["audit"][0]["label_evidence"] == "schema_only"
    assert load_tasks(result["tasks_path"])[0].expected == {"ok": True}


@pytest.mark.parametrize("split", ["tune", "held_out"])
def test_distillation_never_uses_evaluation_splits(tmp_path, split):
    with pytest.raises(ValueError, match="cannot be used"):
        distill(distill_request(rows=[{**rows(1)[0], "split": split}]), tmp_path, teacher())


def test_distillation_unknown_cost_stops_after_one_request(tmp_path):
    calls = []
    result = distill(distill_request(), tmp_path, teacher(cost=None, calls=calls))
    assert result["status"] == "cost_unavailable" and len(calls) == 1
    assert result["accepted_rows"] == 0


def test_distillation_cost_overrun_stops_and_is_not_hidden(tmp_path):
    calls = []
    result = distill(distill_request(), tmp_path, teacher(cost=0.01, calls=calls))
    assert result["status"] == "budget_exceeded" and result["accounted_cost_usd"] == 0.01
    assert len(calls) == 1 and result["accepted_rows"] == 0


def test_distillation_zero_cost_local_teacher(tmp_path):
    result = distill(distill_request(max_cost_usd=0), tmp_path, teacher(cost=0))
    assert result["requests_attempted"] == 2 and result["accepted_rows"] == 2


def test_distillation_cancel_and_failure_do_not_retry(tmp_path):
    event = threading.Event()
    event.set()
    calls = []
    result = distill(distill_request(), tmp_path / "cancel", teacher(calls=calls), cancel=event)
    assert result["status"] == "cancelled" and calls == []

    def failing(**kwargs):
        raise RuntimeError("sensitive provider body must not be persisted")

    result = distill(distill_request(), tmp_path / "failure", failing)
    assert result["status"] == "failed" and result["requests_attempted"] == 1
    assert "sensitive provider" not in json.dumps(result)
