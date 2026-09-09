"""Reviewed staging repairs and host-scored components executed only in Docker."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .containers import copy_project, run_container
from .intelligence import positive
from .runner import write_json
from .workspace import apply_proposal, code_messages, prepare_proposal, read_context, safe_path

_CONFIG = {
    "conftest.py",
    "pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
    "setup.py",
    "package.json",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "cargo.toml",
    "cargo.lock",
    "build.rs",
    "unittest.py",
    "sitecustomize.py",
    "usercustomize.py",
}
_REPORT = "llm-acceptance-report.json"
_JUNIT = "llm-acceptance-report.xml"

# Imported with -I before exposing the project to Python's import machinery.
# The command comes from the host; generated source cannot replace this runner.
_UNITTEST_RUNNER = r"""
import io,json,os,sys,unittest
from pathlib import Path
report=Path('/workspace/llm-acceptance-report.json')
files=json.loads(sys.argv[1])
sys.path.insert(0,'/workspace')
suite=unittest.TestSuite()
loader=unittest.TestLoader()
for name in files:
    module=name[:-3].replace('/','.')
    suite.addTests(loader.loadTestsFromName(module))
result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(suite)
record={'schema_version':1,'runtime':'python','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'expected_failures':len(result.expectedFailures),'unexpected_successes':len(result.unexpectedSuccesses)}
report.write_text(json.dumps(record),encoding='utf-8')
sys.exit(0 if result.wasSuccessful() else 1)
"""

# The candidate receives only one input. Expected labels remain on the host.
# Its stdout is nested as data in a trusted wrapper record, not parsed as metrics.
_COMPONENT_RUNNER = r"""
import json,subprocess,sys,tempfile,time
spec=json.loads(sys.argv[1])
with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
    child=subprocess.Popen(spec['command'],stdin=subprocess.PIPE,stdout=output,stderr=errors)
    payload=(json.dumps(spec['input'],allow_nan=False)+'\n').encode()
    timed_out=False
    try: child.communicate(payload,timeout=spec['case_timeout_s'])
    except subprocess.TimeoutExpired:
        timed_out=True;child.kill();child.communicate()
    output.seek(0,2); output_size=output.tell(); output.seek(0)
    errors.seek(0,2); error_size=errors.tell(); errors.seek(0)
    record={'schema_version':1,'id':spec['id'],'repeat':spec['repeat'],
            'stdout':output.read(65537).decode('utf-8',errors='replace'),
            'stderr':errors.read(16384).decode('utf-8',errors='replace'),
            'exit_code':child.returncode,'timed_out':timed_out,
            'output_limit_exceeded':output_size>65536 or error_size>65536}
    print(json.dumps(record,allow_nan=False))
"""


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hashes(root):
    return {
        p.relative_to(root).as_posix(): _hash(p)
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
    }


def _protected(name):
    parts = Path(name).parts
    filename = parts[-1].lower()
    return (
        any(
            p.lower()
            in ("tests", "test", "__tests__", "fixtures", "harness", "acceptance", "unittest")
            for p in parts
        )
        or filename in _CONFIG
        or filename.startswith(("test_", "test.", "llm-acceptance-", "component-harness"))
        or any(marker in filename for marker in (".test.", ".spec."))
    )


def _output(project, output_dir):
    root, output = Path(project).resolve(), Path(output_dir).resolve()
    if not root.is_dir():
        raise ValueError("project directory does not exist")
    if output.is_relative_to(root):
        raise ValueError("output directory must be outside the source project")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    return root, output


def _resources(request):
    memory = request.get("memory_mib", 512)
    positive(memory, "memory MiB", 65536)
    if type(memory) is not int or memory < 64:
        raise ValueError("memory MiB must be an integer >=64")
    cpus = positive(request.get("cpus", 1), "CPUs", 64)
    timeout = positive(request.get("timeout_s", 60), "timeout", 3600)
    pull = request.get("pull", False)
    if type(pull) is not bool:
        raise ValueError("pull must be boolean")
    return {
        "memory_mib": memory,
        "cpus": cpus,
        "timeout_s": timeout,
        "pull": pull,
        "network": False,
    }


def _clean_exit(result):
    return result.get("exit_code") == 0 and not any(
        result.get(k, False) for k in ("timed_out", "cancelled", "output_limit_exceeded")
    )


def _acceptance_files(manifest, runtime):
    if runtime == "python":
        return sorted(
            name for name in manifest if name.endswith(".py") and Path(name).name.startswith("test")
        )
    return sorted(
        name
        for name in manifest
        if name.endswith((".js", ".mjs", ".cjs"))
        and (_protected(name) and ("test" in name.lower() or ".spec." in name.lower()))
    )


def _report_bytes(result, filename):
    root = Path(result["artifact_dir"]).resolve()
    path = root / filename
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("structured acceptance report missing, unsafe or oversized")
    return path.read_bytes()


def _acceptance_report(result, runtime):
    if runtime == "python":
        data = json.loads(_report_bytes(result, _REPORT))
        keys = {
            "schema_version",
            "runtime",
            "tests_run",
            "failures",
            "errors",
            "skipped",
            "expected_failures",
            "unexpected_successes",
        }
        if (
            not isinstance(data, dict)
            or set(data) != keys
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or data["runtime"] != "python"
        ):
            raise ValueError("invalid Python acceptance report schema")
        for key in keys - {"schema_version", "runtime"}:
            if type(data[key]) is not int or not 0 <= data[key] <= 100000:
                raise ValueError("invalid Python acceptance counters")
        if any(
            data[k] > data["tests_run"]
            for k in ("failures", "errors", "skipped", "expected_failures", "unexpected_successes")
        ):
            raise ValueError("acceptance counters exceed tests run")
    else:
        xml = _report_bytes(result, _JUNIT)
        if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
            raise ValueError("unsupported JUnit declarations")
        tree = ET.fromstring(xml)
        if tree.tag not in ("testsuites", "testsuite"):
            raise ValueError("invalid JUnit root")
        cases = list(tree.iter("testcase"))
        data = {
            "schema_version": 1,
            "runtime": "node",
            "tests_run": len(cases),
            "failures": sum(c.find("failure") is not None for c in cases),
            "errors": sum(c.find("error") is not None for c in cases),
            "skipped": sum(c.find("skipped") is not None for c in cases),
            "expected_failures": 0,
            "unexpected_successes": 0,
        }
    data["passed"] = (
        _clean_exit(result)
        and data["tests_run"] - data["skipped"] - data["expected_failures"] > 0
        and not any(data[k] for k in ("failures", "errors", "unexpected_successes"))
    )
    data["source"] = (
        "fixed unittest structured report"
        if runtime == "python"
        else "Node built-in JUnit reporter"
    )
    return data


def test_count(result, runtime):
    """Count host-validated structured evidence only; console messages are never evidence."""
    data = result.get("structured_test_report", {})
    count = data.get("tests_run", 0)
    return count if data.get("runtime") == runtime and type(count) is int and count >= 0 else 0


def _test_stage(stage, output, runtime, protected, acceptance, resources, cancel):
    if not acceptance:
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": "No original acceptance tests were supplied.",
            "structured_test_report": {"runtime": runtime, "tests_run": 0, "passed": False},
            "acceptance_intact": True,
        }
    if runtime == "python":
        command = ["python", "-I", "-c", _UNITTEST_RUNNER, json.dumps(acceptance)]
    else:
        command = [
            "node",
            "--test",
            "--test-reporter=junit",
            "--test-reporter-destination=" + _JUNIT,
            *acceptance,
        ]
    result = run_container(
        stage, output, runtime=runtime, command=command, cancel=cancel, **resources
    )
    artifact = Path(result["artifact_dir"]).resolve()
    if artifact != (Path(output).resolve() / "workspace"):
        raise ValueError("unexpected container artifact directory")
    actual = _hashes(artifact)
    intact = all(actual.get(name) == before for name, before in protected.items())
    result["acceptance_intact"] = intact
    try:
        report = _acceptance_report(result, runtime)
        report["passed"] = report["passed"] and intact
        result["structured_test_report"] = report
    except (ValueError, OSError, ET.ParseError) as exc:
        result["structured_test_report"] = {
            "runtime": runtime,
            "tests_run": 0,
            "passed": False,
            "error": str(exc),
        }
    return result


def _generation_cost(generated):
    completion = generated.get("completion", {})
    for source, value in (
        ("provider_reported", completion.get("provider_reported_cost_usd")),
        ("token_accounted", completion.get("accounted_cost_usd")),
        ("route_estimated", generated.get("route", {}).get("estimated_cost_usd")),
    ):
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("invalid generation cost")
            return float(value), source
    return None, "unknown"


def _json_safe(value):
    """Keep malformed cost evidence persistable without inventing a numeric value."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_number": str(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unsupported_response_type": type(value).__name__}


def run_feedback(project, output_dir, request, generate, *, cancel=None, progress=None):
    """Repair staging only; return a reviewable proposal and a durable partial cost ledger."""
    if request.get("reviewed_execution") is not True:
        raise ValueError(
            "review and enable execution of generated changes in the disposable container"
        )
    root, output = _output(project, output_dir)
    steps = positive(request.get("max_iterations", 3), "iterations", 10)
    if type(steps) is not int:
        raise ValueError("iterations must be an integer")
    budget = request.get("max_cost_usd", 0.1)
    if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 <= budget <= 100:
        raise ValueError("cost budget must be finite and in [0,100]")
    max_tokens = positive(request.get("max_tokens", 2048), "output tokens", 32768)
    if type(max_tokens) is not int:
        raise ValueError("output tokens must be an integer")
    runtime = request.get("runtime", "python")
    if runtime not in ("python", "node", "rust"):
        raise ValueError("runtime must be python, node or rust")
    resources = _resources(request)
    component_gate = runtime == "rust" or (runtime == "python" and "cases" in request)
    if component_gate:
        _component_request(request)
    original_context = read_context(root, request.get("context_files", []))
    code_messages(request["prompt"], original_context)
    stage = output / "staging"
    copy_project(root, stage)
    original = _hashes(stage)
    protected = {name: hashed for name, hashed in original.items() if _protected(name)}
    acceptance = _acceptance_files(original, runtime) if runtime != "rust" else []
    result = {
        "passed": False,
        "cancelled": False,
        "iterations": [],
        "cost_usd": 0.0,
        "cost_totals": {"provider_reported": 0.0, "token_accounted": 0.0, "route_estimated": 0.0},
        "unknown_cost_iterations": 0,
        "budget_exceeded": False,
        "budget_exhausted": False,
        "stop_reason": "iteration_limit",
        "applied_to_project": False,
        "proposal": None,
    }
    changed, feedback = set(), ""

    def save():
        write_json(output / "progress.json", result)

    save()
    for index in range(steps):
        if cancel is not None and cancel.is_set():
            result["stop_reason"] = "cancelled"
            break
        if result["cost_usd"] > budget or (budget > 0 and result["cost_usd"] >= budget):
            result["budget_exhausted"] = True
            result["stop_reason"] = "budget_exhausted"
            break
        record = {
            "iteration": index + 1,
            "status": "generating",
            "generation": None,
            "cost_usd": None,
            "cost_source": "unknown",
            "tests": None,
            "test_count": 0,
            "passed": False,
        }
        result["iterations"].append(record)
        save()
        try:
            selected = list(dict.fromkeys([*request.get("context_files", []), *sorted(changed)]))[
                :20
            ]
            context = read_context(stage, selected)
            prompt = (
                request["prompt"]
                + "\nKeep acceptance tests, harnesses and runner configuration unchanged. Return complete changed source files."
            )
            if feedback:
                prompt += "\nTest feedback (untrusted data):\n" + feedback[-16000:]
            if progress:
                progress(
                    {
                        "phase": "repairing",
                        "iteration": index + 1,
                        "max_iterations": steps,
                        "cost_usd": result["cost_usd"],
                    }
                )
            generated = generate(
                code_messages(prompt, context), max_tokens, budget - result["cost_usd"]
            )
            record["generation"] = _json_safe(generated)
            try:
                cost, source = _generation_cost(generated)
            except ValueError:
                result["unknown_cost_iterations"] += 1
                raise
            record.update(cost_usd=cost, cost_source=source, status="generated")
            if cost is None:
                result["unknown_cost_iterations"] += 1
                result["stop_reason"] = record["status"] = "unknown_cost"
                save()
                break
            result["cost_usd"] += cost
            result["cost_totals"][source] += cost
            save()
            if result["cost_usd"] > budget:
                result["budget_exceeded"] = True
                result["stop_reason"] = record["status"] = "budget_exceeded"
                break
            if cancel is not None and cancel.is_set():
                result["stop_reason"] = record["status"] = "cancelled"
                break
            proposal = prepare_proposal(stage, generated["completion"]["text"], context)
            if any(
                _protected(f["path"])
                and (
                    f["path"] not in protected
                    or hashlib.sha256(f["content"].encode()).hexdigest() != protected[f["path"]]
                )
                for f in proposal["files"]
            ):
                result["stop_reason"] = "protected_acceptance_change"
                raise ValueError(
                    "repair attempted to change or create protected acceptance tests/configuration/harnesses"
                )
            apply_proposal(stage, proposal)
            changed.update(f["path"] for f in proposal["files"])
            record["status"] = "testing"
            save()
            if component_gate:
                tested = run_component(stage, output / f"test-{index + 1}", request, cancel=cancel)
                count = (
                    len(tested.get("measurements", {}).get("rows", []))
                    if tested.get("measurements")
                    else 0
                )
                passed = tested["passed"]
            else:
                tested = _test_stage(
                    stage,
                    output / f"test-{index + 1}",
                    runtime,
                    protected,
                    acceptance,
                    resources,
                    cancel,
                )
                count = test_count(tested, runtime)
                passed = tested["structured_test_report"]["passed"]
            record.update(
                tests=tested,
                test_count=count,
                passed=passed,
                status="passed" if passed else "failed",
            )
            save()
            if passed:
                result["passed"] = True
                result["stop_reason"] = "acceptance_passed"
                break
            feedback = json.dumps({"test_count": count, "result": tested}, allow_nan=False)
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            if result["stop_reason"] == "iteration_limit":
                result["stop_reason"] = "iteration_error"
            save()
            break
    result["cancelled"] = bool(cancel is not None and cancel.is_set())
    if result["cancelled"]:
        result["passed"] = False
        result["stop_reason"] = "cancelled"
    if not result["passed"] and (
        result["cost_usd"] > budget or (budget > 0 and result["cost_usd"] >= budget)
    ):
        result["budget_exhausted"] = True
        if result["stop_reason"] == "iteration_limit":
            result["stop_reason"] = "budget_exhausted"
    try:
        for name in changed:
            path = safe_path(root, name)
            current = _hash(path) if path.exists() else None
            if current != original.get(name):
                raise ValueError("original project changed during repair; regenerate the proposal")
        files = [
            {"path": name, "content": safe_path(stage, name).read_text(encoding="utf-8")}
            for name in sorted(changed)
        ]
        result["proposal"] = prepare_proposal(
            root,
            json.dumps(
                {
                    "summary": f"Repair loop: {result['stop_reason']} after {len(result['iterations'])} recorded iterations. Review before applying.",
                    "files": files,
                }
            ),
            original_context,
        )
    except Exception as exc:
        result["passed"] = False
        result["stop_reason"] = "stale_original"
        result["proposal_error"] = f"{type(exc).__name__}: {exc}"
    result["cost_note"] = (
        "Budget accounting combines explicitly labelled provider-reported, token-accounted and route-estimated values; unknown costs are not zero."
    )
    save()
    write_json(output / "result.json", result)
    return result


def _component_request(request):
    runtime = request.get("runtime", "python")
    if runtime not in ("python", "rust"):
        raise ValueError("deterministic components support Python or Rust")
    cases = request.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError("provide 1–100 component acceptance cases")
    for case in cases:
        if (
            not isinstance(case, dict)
            or set(case) != {"id", "input", "expected"}
            or not isinstance(case["id"], str)
            or not case["id"]
        ):
            raise ValueError("component cases require a nonempty string id, input and expected")
        if len(json.dumps(case, allow_nan=False).encode()) > 65536:
            raise ValueError("component case exceeds 64 KiB")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("component case IDs must be unique")
    repeats = request.get("repeats", 3)
    if type(repeats) is not int or not 1 <= repeats <= 20 or len(cases) * repeats > 500:
        raise ValueError("component evaluation exceeds repeat/request limit")
    threshold = request.get("min_quality", 1.0)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (float, int))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("minimum quality must be 0–1")
    timeout = positive(
        request.get("case_timeout_s", min(30, request.get("timeout_s", 60))), "case timeout", 30
    )
    duration = positive(request.get("max_duration_s", 300), "component duration", 3600)
    return runtime, cases, repeats, threshold, timeout, duration


def _component_row(container, case, repeat):
    row = {
        "id": case["id"],
        "repeat": repeat,
        "passed": False,
        "actual": None,
        "exit_code": container.get("exit_code"),
        "timed_out": container.get("timed_out", False),
        "cancelled": container.get("cancelled", False),
        "latency_s": container.get("elapsed_s"),
    }
    if not _clean_exit(container):
        row["error"] = "container execution failed or was interrupted"
        return row
    try:
        data = json.loads(container["stdout"])
        keys = {
            "schema_version",
            "id",
            "repeat",
            "stdout",
            "stderr",
            "exit_code",
            "timed_out",
            "output_limit_exceeded",
        }
        if (
            not isinstance(data, dict)
            or set(data) != keys
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or data["id"] != case["id"]
            or type(data["repeat"]) is not int
            or data["repeat"] != repeat
        ):
            raise ValueError("invalid case output schema or row identity")
        if (
            type(data["exit_code"]) is not int
            or type(data["timed_out"]) is not bool
            or type(data["output_limit_exceeded"]) is not bool
            or not isinstance(data["stdout"], str)
            or not isinstance(data["stderr"], str)
        ):
            raise ValueError("invalid case output field types")
        row.update(
            exit_code=data["exit_code"],
            timed_out=data["timed_out"],
            output_limit_exceeded=data["output_limit_exceeded"],
        )
        if data["timed_out"] or data["exit_code"] != 0 or data["output_limit_exceeded"]:
            row["error"] = "component timeout, output limit or nonzero exit"
            return row
        actual = json.loads(
            data["stdout"],
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite JSON")),
        )
        row["actual"] = actual
        row["passed"] = json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            case["expected"], sort_keys=True, allow_nan=False
        )
    except (ValueError, TypeError, KeyError) as exc:
        row["error"] = str(exc)
    return row


def run_component(project, output_dir, request, *, cancel=None):
    """Compile/run only in Docker; score expected labels and validate each row on the host."""
    if request.get("reviewed_execution") is not True:
        raise ValueError("component source must be reviewed before container execution")
    runtime, cases, repeats, threshold, case_timeout, duration = _component_request(request)
    resources = _resources(request)
    root, output = _output(project, output_dir)
    entry = request.get("entrypoint", "main.py" if runtime == "python" else "main.rs")
    if not safe_path(root, entry).is_file():
        raise ValueError("component entrypoint is missing")
    stage = output / "source"
    copy_project(root, stage)
    source_hashes = _hashes(stage)
    result = {
        "passed": False,
        "runtime": runtime,
        "source_hashes": source_hashes,
        "build": None,
        "containers": [],
        "container": None,
        "measurements": None,
        "provider_calls": 0,
        "provider_cost_usd": 0,
        "cancelled": False,
        "complete": False,
        "deadline_exceeded": False,
        "note": "Expected labels scored on host. Each case uses a fresh Docker container. No LLM or host execution of candidate code.",
    }
    rows = []
    started = time.monotonic()

    def save():
        result["measurements"] = {
            "rows": rows,
            "quality": sum(r["passed"] for r in rows) / (len(cases) * repeats),
            "requests_completed": len(rows),
            "requests_planned": len(cases) * repeats,
            "latency_p50_s": statistics.median(
                r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float))
            )
            if rows and any(isinstance(r.get("latency_s"), (int, float)) for r in rows)
            else None,
            "child_rss_highwater_kib": None,
            "measurement": "Host-observed Docker command latency including container startup/teardown; no candidate-supplied timing or RSS is trusted.",
        }
        write_json(output / "result.json", result)

    save()
    try:
        if runtime == "rust":
            compiled = run_container(
                stage,
                output / "compile",
                runtime="rust",
                action="build",
                command=["rustc", "-O", str(Path("/workspace") / entry), "-o", "component-program"],
                cancel=cancel,
                **resources,
            )
            result["build"] = compiled
            if not _clean_exit(compiled):
                result["stop_reason"] = "compile_failed"
                save()
                return result
            stage = Path(compiled["artifact_dir"]).resolve()
            if (
                stage != output / "compile/workspace"
                or (stage / "component-program").is_symlink()
                or not (stage / "component-program").is_file()
            ):
                raise ValueError("compiled artifact is missing or unsafe")
            result["binary_sha256"] = _hash(stage / "component-program")
        command = (
            [
                "python",
                "-I",
                "-c",
                "import runpy,sys;sys.path.insert(0,'/workspace');runpy.run_path(sys.argv[1],run_name='__main__')",
                str(Path("/workspace") / entry),
            ]
            if runtime == "python"
            else ["./component-program"]
        )
        for case in cases:
            for repeat in range(repeats):
                if cancel is not None and cancel.is_set():
                    result["cancelled"] = True
                    break
                remaining = duration - (time.monotonic() - started)
                if remaining <= 0:
                    result["deadline_exceeded"] = True
                    break
                spec = {
                    "id": case["id"],
                    "repeat": repeat,
                    "input": case["input"],
                    "command": command,
                    "case_timeout_s": min(case_timeout, remaining),
                }
                options = {**resources, "timeout_s": min(resources["timeout_s"], remaining)}
                container = run_container(
                    stage,
                    output / f"case-{len(rows):03d}",
                    runtime="python",
                    command=[
                        "python",
                        "-I",
                        "-c",
                        _COMPONENT_RUNNER,
                        json.dumps(spec, allow_nan=False),
                    ],
                    cancel=cancel,
                    **options,
                )
                result["containers"].append(container)
                result["container"] = container
                rows.append(_component_row(container, case, repeat))
                save()
            if result["cancelled"] or result["deadline_exceeded"]:
                break
        result["complete"] = (
            len(rows) == len(cases) * repeats
            and not result["cancelled"]
            and not result["deadline_exceeded"]
        )
        result["passed"] = result["complete"] and result["measurements"]["quality"] >= threshold
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["cancelled"] = result["cancelled"] or bool(cancel is not None and cancel.is_set())
        result["complete"] = result["complete"] and not result["cancelled"]
        result["passed"] = result["passed"] and not result["cancelled"]
        result["elapsed_s"] = time.monotonic() - started
        save()
    return result
