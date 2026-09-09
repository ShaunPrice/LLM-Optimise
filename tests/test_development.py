import json
from pathlib import Path

import pytest

from llm_optimise import development
from llm_optimise.containers import copy_project
from llm_optimise.development import run_component, run_feedback


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("def double(x):\n    return x\n")
    (root / "tests").mkdir()
    (root / "tests/test_main.py").write_text(
        "import unittest\nfrom main import double\nclass Acceptance(unittest.TestCase):\n    def test_double(self): self.assertEqual(double(3),6)\n"
    )
    (root / "tests/fixture.bin").write_bytes(b"\xff\x00\x80")
    (root / "pyproject.toml").write_text('[project]\nname="fixture"\n')
    return root


def generation(files=None, cost=0, source="accounted_cost_usd"):
    completion = {
        "text": json.dumps({"summary": "fixture generation, no model called", "files": files or []})
    }
    if source:
        completion[source] = cost
    return {"completion": completion, "route": {"estimated_cost_usd": None}}


def request(**changes):
    return {
        "reviewed_execution": True,
        "prompt": "Correct double(x).",
        "context_files": ["main.py"],
        "max_iterations": 2,
        "max_cost_usd": 0.1,
        **changes,
    }


def fake_container(source, output, *, runtime="python", **kwargs):
    artifact = Path(output) / "workspace"
    copy_project(source, artifact)
    correct = "return x * 2" in (artifact / "main.py").read_text()
    report = {
        "schema_version": 1,
        "runtime": "python",
        "tests_run": 1,
        "failures": 0 if correct else 1,
        "errors": 0,
        "skipped": 0,
        "expected_failures": 0,
        "unexpected_successes": 0,
    }
    (artifact / development._REPORT).write_text(json.dumps(report))
    assert kwargs["network"] is False
    assert kwargs["command"][:3] == ["python", "-I", "-c"]
    return {
        "artifact_dir": str(artifact),
        "exit_code": 0 if correct else 1,
        "stdout": "Ran 500 tests\nOK",
        "stderr": "",
        "timed_out": False,
        "cancelled": False,
        "output_limit_exceeded": False,
        "elapsed_s": 0.01,
    }


def test_console_test_counts_are_never_a_pass_gate():
    for runtime, text in [
        ("python", "Ran 5 tests\nOK"),
        ("node", "# tests 5"),
        ("rust", "test result: ok. 5 passed"),
    ]:
        assert development.test_count({"stdout": text, "stderr": "", "exit_code": 0}, runtime) == 0


def test_staged_repair_retains_failures_binary_fixtures_and_cost_sources(
    project, tmp_path, monkeypatch
):
    monkeypatch.setattr(development, "run_container", fake_container)
    calls = []

    def generate(messages, tokens, remaining):
        calls.append((messages, remaining))
        content = (
            "def double(x):\n    return x\n"
            if len(calls) == 1
            else "def double(x):\n    return x * 2\n"
        )
        return generation([{"path": "main.py", "content": content}], 0.01)

    result = run_feedback(project, tmp_path / "repair", request(), generate)
    assert result["passed"] and len(result["iterations"]) == 2
    assert result["iterations"][0]["passed"] is False
    assert result["iterations"][1]["test_count"] == 1
    assert result["cost_usd"] == 0.02 and result["cost_totals"]["token_accounted"] == 0.02
    assert calls[1][1] == pytest.approx(0.09)
    assert (project / "main.py").read_text().endswith("return x\n")
    assert (tmp_path / "repair/staging/tests/fixture.bin").read_bytes() == b"\xff\x00\x80"
    assert result["proposal"]["files"][0]["path"] == "main.py"


@pytest.mark.parametrize(
    "name",
    [
        "tests/test_main.py",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "build.rs",
        "component-harness.py",
        "llm-acceptance-report.json",
        "conftest.py",
    ],
)
def test_acceptance_config_and_harness_changes_are_rejected_and_recorded(project, tmp_path, name):
    result = run_feedback(
        project,
        tmp_path / "protected",
        request(),
        lambda *args: generation([{"path": name, "content": "tampered"}], 0.02),
    )
    assert not result["passed"]
    assert result["stop_reason"] == "protected_acceptance_change"
    assert result["cost_usd"] == 0.02
    assert result["iterations"][0]["generation"] is not None
    assert result["iterations"][0]["tests"] is None
    assert json.loads((tmp_path / "protected/result.json").read_text())["iterations"][0]["error"]


def test_budget_breach_records_paid_response_without_applying_or_testing(
    project, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        development,
        "run_container",
        lambda *a, **k: pytest.fail("must not execute after budget breach"),
    )
    result = run_feedback(
        project,
        tmp_path / "budget",
        request(),
        lambda *a: generation(
            [{"path": "main.py", "content": "changed"}], 0.2, "provider_reported_cost_usd"
        ),
    )
    assert result["budget_exceeded"] and result["cost_usd"] == 0.2
    assert result["cost_totals"]["provider_reported"] == 0.2
    assert result["iterations"][0]["status"] == "budget_exceeded"
    assert not result["proposal"]["files"]


def test_unknown_cost_and_callback_failure_preserve_partial_iterations(project, tmp_path):
    result = run_feedback(
        project, tmp_path / "unknown", request(), lambda *a: generation(source=None)
    )
    assert result["stop_reason"] == "unknown_cost" and result["unknown_cost_iterations"] == 1
    assert result["iterations"][0]["generation"]

    def failed(*args):
        raise RuntimeError("preflight blocked")

    failed_result = run_feedback(project, tmp_path / "callback-error", request(), failed)
    assert len(failed_result["iterations"]) == 1
    assert failed_result["iterations"][0]["error"].endswith("preflight blocked")


def test_route_estimates_are_labelled(project, tmp_path):
    def estimated(*args):
        result = generation(source=None)
        result["route"]["estimated_cost_usd"] = 0.2
        return result

    result = run_feedback(project, tmp_path / "estimated", request(), estimated)
    assert result["cost_totals"]["route_estimated"] == 0.2
    assert result["iterations"][0]["cost_source"] == "route_estimated"


def test_no_tests_cannot_pass_from_printed_success(project, tmp_path, monkeypatch):
    def no_tests(source, output, **kwargs):
        result = fake_container(source, output, **kwargs)
        report = Path(result["artifact_dir"]) / development._REPORT
        data = json.loads(report.read_text())
        data.update(tests_run=0, failures=0)
        report.write_text(json.dumps(data))
        result["exit_code"] = 0
        return result

    monkeypatch.setattr(development, "run_container", no_tests)
    result = run_feedback(
        project, tmp_path / "no-tests", request(max_iterations=1), lambda *a: generation()
    )
    assert not result["passed"] and result["iterations"][0]["test_count"] == 0


def test_execution_tampering_with_acceptance_is_detected(project, tmp_path, monkeypatch):
    def tamper(source, output, **kwargs):
        result = fake_container(source, output, **kwargs)
        (Path(result["artifact_dir"]) / "tests/fixture.bin").write_bytes(b"changed")
        return result

    monkeypatch.setattr(development, "run_container", tamper)
    result = run_feedback(
        project,
        tmp_path / "tamper",
        request(max_iterations=1),
        lambda *a: generation(
            [{"path": "main.py", "content": "def double(x):\n    return x * 2\n"}]
        ),
    )
    assert not result["passed"]
    assert result["iterations"][0]["tests"]["acceptance_intact"] is False


def test_node_junit_uses_cases_not_stdout(tmp_path):
    (tmp_path / development._JUNIT).write_text(
        '<testsuites><testsuite><testcase name="actual"><failure message="bad"/></testcase></testsuite></testsuites>'
    )
    result = {"artifact_dir": str(tmp_path), "stdout": "# tests 500", "exit_code": 0}
    report = development._acceptance_report(result, "node")
    assert report["tests_run"] == 1 and report["failures"] == 1 and not report["passed"]


def component_request(**extra):
    return {
        "reviewed_execution": True,
        "repeats": 1,
        "cases": [
            {"id": "one", "input": 3, "expected": 6},
            {"id": "two", "input": 4, "expected": 8},
        ],
        **extra,
    }


def component_container(source, output, **kwargs):
    spec = json.loads(kwargs["command"][-1])
    assert "expected" not in spec and "cases" not in spec
    assert kwargs["network"] is False
    record = {
        "schema_version": 1,
        "id": spec["id"],
        "repeat": spec["repeat"],
        "stdout": json.dumps(spec["input"] * 2),
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
        "output_limit_exceeded": False,
    }
    return {
        "exit_code": 0,
        "stdout": json.dumps(record),
        "stderr": "",
        "elapsed_s": 0.1,
        "timed_out": False,
        "cancelled": False,
        "output_limit_exceeded": False,
    }


def test_component_labels_stay_host_side_and_binary_hashes_work(project, tmp_path, monkeypatch):
    monkeypatch.setattr(development, "run_container", component_container)
    result = run_component(project, tmp_path / "component", component_request())
    assert result["passed"] and result["measurements"]["quality"] == 1
    assert "tests/fixture.bin" in result["source_hashes"]
    assert result["measurements"]["child_rss_highwater_kib"] is None
    assert result["measurements"]["latency_p50_s"] == 0.1


@pytest.mark.parametrize(
    "changes",
    [
        {"min_quality": 2},
        {"min_quality": float("nan")},
        {"cases": [{"id": "a", "input": 1, "expected": 2}, {"id": "a", "input": 2, "expected": 4}]},
        {"repeats": True},
    ],
)
def test_component_invalid_request_precedes_execution(project, tmp_path, monkeypatch, changes):
    monkeypatch.setattr(
        development, "run_container", lambda *a, **k: pytest.fail("invalid request executed")
    )
    with pytest.raises(ValueError):
        run_component(project, tmp_path / "invalid", component_request(**changes))


def test_output_containment_precedes_any_execution(project):
    with pytest.raises(ValueError, match="outside"):
        run_component(project, project / "output", component_request())
    with pytest.raises(ValueError, match="outside"):
        run_feedback(project, project / "output", request(), lambda *a: generation())


def test_component_timeout_and_malformed_rows_preserve_failures(project, tmp_path, monkeypatch):
    calls = []

    def partial(source, output, **kwargs):
        result = component_container(source, output, **kwargs)
        calls.append(result)
        data = json.loads(result["stdout"])
        if len(calls) == 1:
            data.update(timed_out=True, exit_code=-9)
        result["stdout"] = json.dumps(data)
        return result

    monkeypatch.setattr(development, "run_container", partial)
    result = run_component(project, tmp_path / "partial", component_request())
    assert result["complete"] and not result["passed"]
    assert result["measurements"]["quality"] == 0.5
    assert result["measurements"]["rows"][0]["timed_out"]
    assert result["measurements"]["rows"][1]["passed"]


@pytest.mark.parametrize(
    "field,value", [("id", "forged"), ("repeat", False), ("exit_code", False), ("timed_out", 0)]
)
def test_component_row_schema_is_strict(field, value):
    case = {"id": "one", "input": 3, "expected": 6}
    record = {
        "schema_version": 1,
        "id": "one",
        "repeat": 0,
        "stdout": "6",
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
        "output_limit_exceeded": False,
    }
    record[field] = value
    row = development._component_row(
        {"exit_code": 0, "stdout": json.dumps(record), "elapsed_s": 1}, case, 0
    )
    assert not row["passed"] and "error" in row


def test_rust_repair_requires_host_scored_acceptance_cases(project, tmp_path):
    with pytest.raises(ValueError, match="acceptance cases"):
        run_feedback(project, tmp_path / "rust", request(runtime="rust"), lambda *a: generation())


def test_nonfinite_cost_still_leaves_a_valid_partial_ledger(project, tmp_path):
    result = run_feedback(
        project, tmp_path / "nonfinite", request(), lambda *a: generation(cost=float("nan"))
    )
    assert not result["passed"] and result["unknown_cost_iterations"] == 1
    saved = json.loads((tmp_path / "nonfinite/result.json").read_text())
    assert saved["iterations"][0]["generation"]["completion"]["accounted_cost_usd"] == {
        "invalid_nonfinite_number": "nan"
    }
    assert "invalid generation cost" in saved["iterations"][0]["error"]


def test_python_feedback_can_use_host_scored_component_gate(project, tmp_path, monkeypatch):
    monkeypatch.setattr(development, "run_container", component_container)
    result = run_feedback(
        project,
        tmp_path / "component-repair",
        request(cases=[{"id": "one", "input": 3, "expected": 6}], repeats=1),
        lambda *a: generation(),
    )
    assert result["passed"] and result["iterations"][0]["test_count"] == 1
    assert result["iterations"][0]["tests"]["measurements"]["quality"] == 1


def test_zero_cost_budget_allows_bounded_free_repairs(project, tmp_path, monkeypatch):
    monkeypatch.setattr(development, "run_container", fake_container)
    calls = []

    def generate(messages, tokens, remaining):
        calls.append(remaining)
        return generation([{"path": "main.py", "content": "def double(x):\n    return x * 2\n"}])

    result = run_feedback(project, tmp_path / "free", request(max_cost_usd=0), generate)
    assert result["passed"] and calls == [0] and result["cost_usd"] == 0
    assert not result["budget_exceeded"]
