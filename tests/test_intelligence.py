import dataclasses

import pytest

from llm_optimise import intelligence
from llm_optimise.context import select_context
from llm_optimise.intelligence import Intelligence
from llm_optimise.quality import Task
from llm_optimise.routing import ProviderModel


def model(**changes):
    values = dict(
        id="local",
        model="tiny",
        provider="openai",
        location="local",
        base_url="http://localhost:8181/v1",
        context_window=8192,
        max_output_tokens=2048,
        input_cost_per_million=0,
        output_cost_per_million=0,
        revision="r1",
    )
    return ProviderModel(**(values | changes))


@pytest.fixture
def engine(tmp_path, monkeypatch):
    calls = []

    def complete(m, messages, max_tokens, **options):
        calls.append((m, messages, options))
        return {
            "text": "yes",
            "latency_ms": 42,
            "input_tokens": 32,
            "output_tokens": 1,
            "accounted_cost_usd": 0,
            "provider_reported_cost_usd": None,
        }

    monkeypatch.setattr(intelligence, "complete", complete)
    return Intelligence(tmp_path, hardware_id="fixture"), calls


def test_cache_preserves_constraints_and_does_not_double_count(engine):
    lab, calls = engine
    m = model()
    messages = [{"role": "user", "content": "ok"}]
    first = lab.run([m], messages, cache=True)
    second = lab.run([m], messages, cache=True)
    assert len(calls) == 1 and calls[0][2]["temperature"] == 0
    assert not first["completion"]["cache_hit"] and second["completion"]["cache_hit"]
    assert (
        second["completion"]["accounted_cost_usd"] == 0
        and second["completion"]["response_id"] is None
    )
    assert lab.status()["measurements"][0]["samples"] == 1
    with pytest.raises(ValueError, match="placement"):
        lab.run([m], messages, {"placement": "cloud", "objective": "cost"}, cache=True)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change", [{"revision": "r2"}, {"adapter_revision": "a2"}, {"model": "other"}]
)
def test_cache_invalidated_by_model_and_adapter_versions(engine, change):
    lab, calls = engine
    messages = [{"role": "user", "content": "ok"}]
    lab.run([model()], messages, cache=True)
    lab.run([dataclasses.replace(model(), **change)], messages, cache=True)
    assert len(calls) == 2


def test_cache_key_covers_prompt_schema_and_decoding(engine):
    lab, calls = engine
    m = model()
    for prompt, schema, prefix in [
        ("a", None, False),
        ("b", None, False),
        ("b", {"type": "object"}, False),
        ("b", None, True),
    ]:
        lab.run(
            [m],
            [{"role": "user", "content": prompt}],
            json_schema=schema,
            prefix_cache=prefix,
            cache=True,
        )
    assert len(calls) == 4


def test_cache_bound_and_expiration(tmp_path):
    lab = Intelligence(tmp_path, max_cache_bytes=100)
    lab.cache_put("a", {"text": "a" * 50})
    lab.cache_put("b", {"text": "b" * 50})
    assert lab.status()["cache"]["bytes"] <= 100
    assert lab.cache_get("a") is None
    with lab.connect() as db:
        db.execute("UPDATE cache SET expires=0")
    assert lab.cache_get("b") is None


def test_calibration_scoped_and_unscored_chat_not_quality(engine):
    lab, _ = engine
    m = model()
    messages = [{"role": "user", "content": "yes"}]
    for _ in range(4):
        lab.run([m], messages, task_class="sentiment")
    calibrated, evidence = lab.calibrated_models(
        [m], "sentiment", intelligence.estimate_input_tokens(messages)
    )
    assert calibrated[0].latency_ms == 42 and calibrated[0].quality is None
    assert evidence["local"]["quality_samples"] == 0
    other, _ = lab.calibrated_models(
        [m], "extraction", intelligence.estimate_input_tokens(messages)
    )
    assert other[0].latency_ms is None
    revised, _ = lab.calibrated_models(
        [dataclasses.replace(m, revision="r2")],
        "sentiment",
        intelligence.estimate_input_tokens(messages),
    )
    assert revised[0].latency_ms is None


def test_scored_calibration_uses_lower_confidence_bound(engine):
    lab, _ = engine
    m = model()
    task = Task("t", "yes", "yes")
    result = lab.calibrate([m], [task], task_class="labels", repeats=6, max_requests=6)
    assert len(result["requests"]) == 6
    tokens = intelligence.estimate_input_tokens(
        [{"role": "system", "content": task.system}, {"role": "user", "content": task.prompt}]
    )
    calibrated, evidence = lab.calibrated_models([m], "labels", tokens)
    assert evidence["local"]["quality_mean"] == 1
    assert 0.5 < calibrated[0].quality < 1


def test_calibration_preflight_blocks_excess_calls(engine):
    lab, calls = engine
    with pytest.raises(ValueError, match="request limit"):
        lab.calibrate(
            [model()], [Task("t", "yes", "yes")], task_class="x", repeats=6, max_requests=5
        )
    expensive = model(
        id="cloud",
        location="cloud",
        base_url="https://api.example.com",
        input_cost_per_million=10000,
        output_cost_per_million=10000,
    )
    with pytest.raises(ValueError, match="cost exceeds"):
        lab.calibrate([expensive], [Task("t", "yes", "yes")], task_class="x", max_cost_usd=0.01)
    assert not calls


def test_specialists_reject_wrong_task_class(engine):
    lab, calls = engine
    with pytest.raises(ValueError, match="no models"):
        lab.run(
            [model(task_classes=("labels",))],
            [{"role": "user", "content": "ok"}],
            task_class="coding",
        )
    assert not calls


def test_context_selection_tracks_sources_and_omitted_facts():
    text = "padding\n" * 60 + "Critical alarm threshold is 70.\n" + "padding\n" * 60
    result = select_context(
        "alarm threshold",
        {"sensor.txt": text},
        max_chars=300,
        chunk_lines=5,
        required_facts=["Critical alarm threshold is 70.", "missing fact"],
    )
    assert "threshold is 70" in result["text"]
    assert result["output_chars"] <= 300 and result["output_chars"] < result["input_chars"]
    assert result["required_facts_missing"] == ["missing fact"] and not result["required_fact_gate"]
    assert result["selected"][0]["start_line"] == 61
    assert "sensor.txt" in result["source_hashes"]


def test_cache_never_stores_environment_secrets(engine, monkeypatch):
    lab, _ = engine
    monkeypatch.setenv("MY_KEY", "top-secret-credential")
    lab.run([model(api_key_env="MY_KEY")], [{"role": "user", "content": "yes"}], cache=True)
    assert b"top-secret-credential" not in lab.path.read_bytes()


def test_observation_storage_bound(engine):
    lab, _ = engine
    lab.max_records = 3
    for _ in range(6):
        lab.record(model(), "x", 100, {"latency_ms": 1})
    assert lab.status()["measurements"][0]["samples"] == 3


def test_context_rejects_unsupported_limits():
    with pytest.raises(ValueError):
        select_context("x", {"a": "b"}, max_chars=True)
    with pytest.raises(ValueError):
        select_context("x", {"a": "b"}, required_facts="not a list")


def test_fair_retention_preserves_less_frequent_specialty(engine):
    lab, _ = engine
    lab.max_records = 4
    lab.record(model(), "rare", 100, {"latency_ms": 2}, 1.0)
    for _ in range(12):
        lab.record(model(), "common", 100, {"latency_ms": 1}, 1.0)
    assert lab.measurements(model(), "rare", 100)["samples"] == 1
    assert lab.measurements(model(), "common", 100)["samples"] == 3
    with pytest.raises(ValueError, match="binary"):
        lab.record(model(), "x", 100, {}, 0.5)


def test_cache_requires_revision_and_scores_hits(engine):
    lab, calls = engine
    messages = [{"role": "user", "content": "yes"}]
    with pytest.raises(ValueError, match="revision"):
        lab.run([model(revision="")], messages, cache=True)
    task = Task("a", "yes", "yes")
    lab.run([model()], messages, cache=True, quality_task=task)
    result = lab.run([model()], messages, cache=True, quality_task=task)
    assert result["quality"] == 1 and len(calls) == 1


def test_calibration_keeps_partial_results_on_provider_error(engine, monkeypatch):
    lab, _ = engine
    original = intelligence.complete
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("provider error")
        return original(*args, **kwargs)

    monkeypatch.setattr(intelligence, "complete", flaky)
    result = lab.calibrate([model()], [Task("a", "yes", "yes")], task_class="x", repeats=3)
    assert result["status"] == "failed" and len(result["requests"]) == 1
    assert result["accounted_cost_usd"] == 0 and len(calls) == 2


def test_calibration_retains_last_bill_and_stops_at_budget(engine, monkeypatch):
    lab, _ = engine
    original = intelligence.complete
    calls = []

    def expensive(*args, **kwargs):
        calls.append(1)
        return {**original(*args, **kwargs), "provider_reported_cost_usd": 0.2}

    monkeypatch.setattr(intelligence, "complete", expensive)
    result = lab.calibrate(
        [model()], [Task("a", "yes", "yes")], task_class="x", repeats=3, max_cost_usd=0.1
    )
    assert result["budget_exceeded"] and result["budget_exhausted"]
    assert len(result["requests"]) == 1 and result["accounted_cost_usd"] == 0.2
    assert len(calls) == 1


def test_calibration_schema_validity_affects_quality(engine):
    lab, _ = engine
    task = Task("a", "yes", "yes", json_schema={"type": "object"})
    result = lab.calibrate([model()], [task], task_class="x")
    assert result["requests"][0]["score"] == 0


def test_specialist_contract_abstains_or_explicitly_escalates(engine, monkeypatch):
    lab, calls = engine
    primary, other = model(), model(id="other")
    options = dict(
        policy={"placement": "local", "objective": "cost", "max_cost_usd": 0.01},
        selected_model="local",
    )
    result = lab.respond(
        [primary, other],
        [{"role": "user", "content": "label"}],
        contract={"allowed_outputs": ["no"]},
        **options,
    )
    assert result["abstained"] and result["output"] is None and len(calls) == 1
    original = intelligence.complete

    def fallback(m, *args, **kwargs):
        return {**original(m, *args, **kwargs), "text": "no" if m.id == "other" else "yes"}

    monkeypatch.setattr(intelligence, "complete", fallback)
    result = lab.respond(
        [primary, other],
        [{"role": "user", "content": "label"}],
        contract={"allowed_outputs": ["no"]},
        escalation_model="other",
        **options,
    )
    assert result["accepted"] and result["output"] == "no" and len(result["attempts"]) == 2


def test_zero_budget_distillation_uses_only_known_free_models(engine, tmp_path):
    from llm_optimise.workbench import execute_workbench

    lab, calls = engine
    result = execute_workbench(
        tmp_path,
        "distill",
        {
            "rows": [{"id": "a", "prompt": "yes", "expected": "yes"}],
            "data_authorized": True,
            "max_requests": 1,
            "max_cost_usd": 0,
            "selected_model": "local",
        },
        models=[model()],
        engine=lab,
    )
    assert result["accepted_rows"] == 1 and len(calls) == 1
    paid = model(input_cost_per_million=1, output_cost_per_million=1)
    result = execute_workbench(
        tmp_path,
        "distill",
        {
            "rows": [{"id": "a", "prompt": "yes", "expected": "yes"}],
            "data_authorized": True,
            "max_requests": 1,
            "max_cost_usd": 0,
            "selected_model": "local",
        },
        models=[paid],
        engine=lab,
    )
    assert result["accepted_rows"] == 0 and len(calls) == 1
