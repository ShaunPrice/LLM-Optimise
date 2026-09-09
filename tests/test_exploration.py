"""Workbenches use an actual local fixture process; this is not LLM/hardware evidence."""

import json
import os
import platform
import sys
import textwrap
import threading
import time

import psutil
import pytest

from llm_optimise.backend import llama_command, speculation_counts
from llm_optimise.config import Candidate, parse_experiment
from llm_optimise.exploration import (
    classify_trial,
    kv_context_preset,
    plan_exploration,
    run_exploration,
    runtime_capabilities,
)


def _diagnostics(result, output):
    """Keep fixture failures actionable on CI, including failed pre-metric trials."""
    logs = {}
    for path in sorted(output.rglob("trial-*.log")):
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 4096))
            logs[str(path.relative_to(output))] = stream.read(4096).decode(
                "utf-8", errors="replace"
            )
    return json.dumps(
        {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "pytest_rss_gib": psutil.Process().memory_info().rss / (1024**3),
            "result": result,
            "server_log_tails": logs,
        },
        indent=2,
        default=str,
    )


@pytest.fixture
def lab(tmp_path):
    runtime = tmp_path / "fixture-server"
    runtime.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""
        import json, os, sys, time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from socketserver import TCPServer
        args = sys.argv[1:]
        flags = "--model --gpu-layers --no-kv-offload --no-op-offload --list-devices --device --spec-draft-model --spec-draft-n-max --spec-draft-ngl --spec-draft-device --spec-type"
        if '--help' in args:
            print(flags); sys.exit(0)
        if '--version' in args:
            print('fixture-runtime 1; not an LLM'); sys.exit(0)
        if '--list-devices' in args:
            print('Available devices:\\n  CUDA0: Fixture CUDA (1024 MiB)\\n  MTL0: Fixture Metal (1024 MiB)\\n  Vulkan0: Fixture Vulkan (1024 MiB)\\n  HIP0: Fixture ROCm (1024 MiB)'); sys.exit(0)
        def value(flag):
            return args[args.index(flag) + 1]
        context = int(value('--ctx-size'))
        if context >= 4096:
            print('out of memory: fixture allocation failure', flush=True); sys.exit(1)
        model = value('--model')
        with open(model + '.pids', 'a') as stream:
            stream.write(str(os.getpid()) + '\\n')
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                print('fixture GET ' + self.path, flush=True)
                self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
            def do_POST(self):
                print('fixture POST ' + self.path, flush=True)
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200); self.end_headers()
                if self.path == '/apply-template':
                    self.wfile.write(json.dumps({'prompt': data['messages'][-1]['content']}).encode()); return
                if 'stall' in os.path.basename(model):
                    time.sleep(30); return
                result = 'wrong' if 'bad' in os.path.basename(model) else 'ok'
                self.wfile.write(('data: ' + json.dumps({'content': result, 'stop': False}) + '\\n\\n').encode())
                final = {'content': '', 'stop': True, 'timings': {'predicted_n': 1, 'predicted_per_second': 100, 'prompt_n': 1}}
                if '--spec-draft-model' in args:
                    final.update(draft_n=8, draft_n_accepted=6)
                self.wfile.write(('data: ' + json.dumps(final) + '\\n\\n').encode()); self.wfile.flush()
        class FixtureServer(ThreadingHTTPServer):
            def server_bind(self):
                # HTTPServer.server_bind reverse-resolves even numeric loopback hosts.
                # This local protocol fixture must not depend on the host's DNS service.
                TCPServer.server_bind(self)
                self.server_name, self.server_port = self.server_address[:2]
        server = FixtureServer(('127.0.0.1', int(value('--port'))), Handler)
        print('fixture listening ' + repr(server.server_address) + ' pid=' + str(os.getpid()), flush=True)
        server.serve_forever()
    """)
    )
    runtime.chmod(0o755)
    if os.name == "nt":
        script = runtime.with_suffix(".py")
        runtime.rename(script)
        runtime = runtime.with_suffix(".cmd")
        runtime.write_text(f'@echo off\n"{sys.executable}" "{script}" %*\n')
    model = tmp_path / "good.gguf"
    model.write_bytes(b"fixture only")
    (tmp_path / "bad.gguf").write_bytes(b"fixture only")
    (tmp_path / "draft.gguf").write_bytes(b"fixture only")
    data = tmp_path / "development.jsonl"
    data.write_text(
        "".join(
            json.dumps({"id": f"dev-{i}", "prompt": f"development {i}", "expected": "ok"}) + "\n"
            for i in range(4)
        )
    )
    spec = {
        "mode": "capacity",
        "experiment": {
            "name": "fixture",
            "dataset": str(data),
            "repeats": 1,
            "warmup": 0,
            "max_tokens": 16,
            "startup_timeout_s": 3,
            "timeout_s": 3,
            "max_trials": 16,
            "limits": {"min_quality": 0.9, "min_available_gib": 0.001, "max_rss_gib": 0.25},
            "candidates": [
                {"name": "good", "model": str(model), "executable": str(runtime), "context": 512}
            ],
        },
    }
    return spec, tmp_path, runtime


def test_fixture_loopback_startup_needs_no_reverse_dns(lab):
    spec, root, runtime = lab
    script = runtime.with_suffix(".py") if os.name == "nt" else runtime
    script.write_text(
        script.read_text().replace(
            "args = sys.argv[1:]",
            "import socket\n"
            "socket.getfqdn = lambda *_args: sys.exit('unexpected reverse DNS lookup')\n"
            "args = sys.argv[1:]",
        )
    )
    spec["capacity"] = {"dimensions": {"context": [512]}}
    result = run_exploration(spec, root / "no-dns")
    assert result["selected_candidate"] == "capacity-001", _diagnostics(result, root / "no-dns")
    assert all(
        not psutil.pid_exists(int(pid))
        for pid in (root / "good.gguf.pids").read_text().splitlines()
    )


def test_parse_object_is_portable_and_does_not_mutate(tmp_path):
    raw = {
        "name": "test",
        "dataset": "tasks.jsonl",
        "candidates": [{"name": "a", "model": "m.gguf"}],
    }
    before = json.dumps(raw)
    parsed = parse_experiment(raw, tmp_path)
    assert parsed.dataset == str(tmp_path / "tasks.jsonl")
    assert parsed.candidates[0].model == str(tmp_path / "m.gguf")
    assert json.dumps(raw) == before


@pytest.mark.parametrize("values", [[2, 1], [1, 1], [True], [], [999999]])
def test_capacity_dimensions_are_bounded_and_increasing(lab, values):
    spec, _, _ = lab
    spec["capacity"] = {"dimensions": {"context": values}}
    with pytest.raises(ValueError):
        plan_exploration(spec)


def test_capacity_staircase_does_not_reset_smaller_dimensions(lab):
    spec, _, _ = lab
    spec["capacity"] = {"dimensions": {"context": [512, 1024, 2048], "threads": [1, 2]}}
    plan = plan_exploration(spec)
    assert [(c["context"], c["threads"]) for c in plan["candidates"]] == [
        (512, 1),
        (1024, 2),
        (2048, 2),
    ]
    assert plan["max_worker_evaluations"] == 4
    assert plan["memory_enforcement"].startswith("sampled")


def test_capacity_real_fixture_failure_cleanup_and_recovery(lab):
    spec, root, _ = lab
    spec["capacity"] = {"dimensions": {"context": [512, 4096, 8192]}, "recovery_timeout_s": 1}
    result = run_exploration(spec, root / "capacity")
    assert result["selected_candidate"] == "capacity-001", _diagnostics(result, root / "capacity")
    assert result["boundary"]["failure_class"] == "allocation_failure", _diagnostics(
        result, root / "capacity"
    )
    assert len(result["trials"]) == 2  # Never attempt the higher unsafe point.
    assert result["recovery"]["control_passed"], _diagnostics(result, root / "capacity")
    assert result["recovery"]["control_candidate"] == "capacity-001"
    assert result["complete"]
    pids = [int(p) for p in (root / "good.gguf.pids").read_text().splitlines()]
    assert len(pids) == 2 and len(set(pids)) == 2  # Fresh baseline and recovery processes.
    assert all(not psutil.pid_exists(pid) for pid in pids)
    saved = json.loads((root / "capacity/exploration.json").read_text())
    assert saved["boundary"] == result["boundary"]


def test_halving_uses_nested_common_dev_subsets_and_never_selects_on_heldout(lab):
    spec, root, _ = lab
    spec["mode"] = "halving"
    base = spec["experiment"]["candidates"][0]
    spec["experiment"]["candidates"].append(
        {**base, "name": "bad", "model": str(root / "bad.gguf")}
    )
    heldout = root / "heldout.jsonl"
    heldout.write_text(
        json.dumps(
            {
                "id": "held-1",
                "prompt": "heldout question",
                "expected": "deliberately-wrong-for-fixture",
            }
        )
        + "\n"
    )
    spec["search"] = {"task_budgets": [1, 2], "eta": 2, "heldout_dataset": str(heldout)}
    result = run_exploration(spec, root / "halving")
    assert result["selected_candidate"] == "good", _diagnostics(result, root / "halving")
    assert not result["heldout"]["passed"]
    assert result["heldout"]["used_for_selection"] is False
    assert [s["task_count"] for s in result["stages"]] == [1, 2, 4, 1]
    assert len(result["stages"][0]["trials"]) == 2
    assert result["stages"][0]["promoted"] == ["good"]
    assert all(t["name"] == "good" for s in result["stages"][1:] for t in s["trials"])
    small = [
        json.loads(x)["id"]
        for x in (root / "halving/development-00.jsonl").read_text().splitlines()
    ]
    large = [
        json.loads(x)["id"]
        for x in (root / "halving/development-02.jsonl").read_text().splitlines()
    ]
    assert large[: len(small)] == small
    assert all(not name.startswith("held-") for name in large)


@pytest.mark.parametrize("duplicate", ["id", "prompt"])
def test_heldout_overlap_is_rejected_before_workers(lab, duplicate):
    spec, root, _ = lab
    spec["mode"] = "halving"
    task = {
        "id": "dev-0" if duplicate == "id" else "new",
        "prompt": "development 0" if duplicate == "prompt" else "unseen",
        "expected": "no",
    }
    path = root / "heldout.jsonl"
    path.write_text(json.dumps(task) + "\n")
    spec["search"] = {"heldout_dataset": str(path)}
    with pytest.raises(ValueError, match="overlaps"):
        run_exploration(spec, root / "rejected")
    assert not (root / "good.gguf.pids").exists()


def test_kv_preset_validates_combinations_and_executes(lab):
    spec, root, _ = lab
    generated = kv_context_preset(spec["experiment"], [512, 1024], ["f16", "q8_0"])
    assert len(generated["candidates"]) == 4
    assert generated["candidates"][1]["flash_attention"] == "on"
    parse_experiment(generated)
    spec.update(mode="kv", kv={"contexts": [512], "cache_types": ["f16", "q8_0"]})
    result = run_exploration(spec, root / "kv")
    assert len(result["trials"]) == 2
    assert all(t["eligible"] for t in result["trials"]), _diagnostics(result, root / "kv")


def test_installed_runtime_probe_and_speculative_measured_counters(lab):
    spec, root, runtime = lab
    cap = runtime_capabilities(runtime)
    assert cap["installed"] and "fixture-runtime" in cap["version"]
    assert set(cap["supported_accelerators"]) == {"cpu", "cuda", "metal", "vulkan", "rocm"}
    cap["flags"].clear()
    assert runtime_capabilities(runtime)["flags"]  # Cached evidence cannot be mutated by callers.
    spec.update(
        mode="speculative",
        speculative={"draft_models": [str(root / "draft.gguf")], "draft_max": [4]},
    )
    result = run_exploration(spec, root / "speculation")
    assert result["trials"] and all(t["eligible"] for t in result["trials"]), _diagnostics(
        result, root / "speculation"
    )
    baseline = next(t for t in result["trials"] if t["name"] == "spec-baseline")
    draft = next(t for t in result["trials"] if t["name"] != "spec-baseline")
    assert baseline["metrics"]["draft_acceptance_rate"] is None
    assert draft["metrics"]["draft_tokens"] == 32
    assert draft["metrics"]["accepted_draft_tokens"] == 24
    assert draft["metrics"]["draft_acceptance_rate"] == 0.75
    assert draft["speedup_vs_baseline"] > 0
    assert "--spec-draft-device" in draft["command"]
    assert draft["command"][draft["command"].index("--spec-draft-device") + 1] == "none"


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"draft_n": 0},
        {"draft_n": 2, "draft_n_accepted": 3},
        {"draft_n": True, "draft_n_accepted": 1},
    ],
)
def test_speculation_never_fabricates_counters(event):
    assert speculation_counts(event) == (None, None, None)


def test_accelerator_comparison_uses_reported_device_not_filename(lab):
    spec, root, runtime = lab
    spec.update(
        mode="accelerators",
        accelerators=[
            {"accelerator": "cpu", "executable": str(runtime)},
            {"accelerator": "cuda", "executable": str(runtime)},
            {"accelerator": "rocm", "executable": str(root / "nonexistent-rocm-server")},
        ],
    )
    plan = plan_exploration(spec)
    assert [c["accelerator"] for c in plan["candidates"]] == ["cpu", "cuda"]
    assert len(plan["excluded_candidates"]) == 1
    candidate = Candidate(**plan["candidates"][1])
    command = llama_command(candidate, 8888, "ephemeral")
    assert command[command.index("--device") + 1] == "CUDA0"
    result = run_exploration(spec, root / "accelerators")
    assert len(result["trials"]) == 2
    assert all(t["eligible"] for t in result["trials"]), _diagnostics(result, root / "accelerators")
    cuda = next(t for t in result["trials"] if t["candidate"]["accelerator"] == "cuda")
    # Enumeration fixture establishes API behaviour, not real accelerator telemetry.
    assert cuda["memory"]["gpu_peak_gib"] is None


def test_accelerator_unverified_and_arbitrary_device_are_rejected(lab):
    spec, root, runtime = lab
    spec.update(
        mode="accelerators",
        accelerators=[{"accelerator": "metal", "executable": str(runtime), "device": "CUDA0"}],
    )
    with pytest.raises(ValueError, match="device"):
        plan_exploration(spec)
    spec["accelerators"] = [{"accelerator": "metal", "executable": str(root / "absent")}]
    result = run_exploration(spec, root / "unavailable")
    assert result["boundary"]["reason"] == "no_verified_runtime"
    assert not result["complete"] and not result["trials"]


def test_cancellation_terminates_stalled_owned_worker(lab):
    spec, root, _ = lab
    stalled = root / "stall.gguf"
    stalled.write_bytes(b"fixture")
    spec["experiment"]["candidates"][0]["model"] = str(stalled)
    spec["experiment"]["timeout_s"] = 20
    spec["capacity"] = {"dimensions": {"context": [512]}}
    spec["max_duration_s"] = 0.6
    started = time.monotonic()
    result = run_exploration(spec, root / "cancel")
    assert result["deadline_exceeded"] and result["cancelled"] and not result["complete"]
    assert time.monotonic() - started < 8
    assert all(
        not psutil.pid_exists(int(pid))
        for pid in (root / "stall.gguf.pids").read_text().splitlines()
    )


def test_pre_cancelled_never_launches_model(lab):
    spec, root, _ = lab
    event = threading.Event()
    event.set()
    result = run_exploration(spec, root / "pre-cancel", cancel=event)
    assert result["cancelled"] and result["trials"] == []
    assert not (root / "good.gguf.pids").exists()


def test_headroom_failure_is_classified_and_never_called_oom():
    assert classify_trial({"status": "failed", "error": "process exited -9"}) == "runtime_failure"
    assert (
        classify_trial({"status": "failed", "error": "insufficient available RAM before launch"})
        == "memory_guard"
    )
    assert (
        classify_trial(
            {
                "status": "complete",
                "eligible": False,
                "rejection_reasons": ["GPU memory unavailable"],
            }
        )
        == "telemetry_unavailable"
    )


def test_exploration_rejects_external_endpoints_and_unbounded_work(lab):
    spec, _, _ = lab
    spec["experiment"]["candidates"][0].update(
        backend="openai", endpoint="http://127.0.0.1:8000/v1"
    )
    with pytest.raises(ValueError, match="managed local"):
        plan_exploration(spec)
    spec["experiment"]["candidates"][0]["backend"] = "llama.cpp"
    spec["experiment"]["limits"]["min_available_gib"] = 0
    with pytest.raises(ValueError, match="headroom"):
        plan_exploration(spec)


def test_existing_output_is_not_overwritten(lab):
    spec, root, _ = lab
    output = root / "existing"
    output.mkdir()
    artifact = output / "keep.txt"
    artifact.write_text("original")
    with pytest.raises(ValueError, match="new or empty"):
        run_exploration(spec, output)
    assert artifact.read_text() == "original"


def test_model_change_between_stages_stops_search(lab):
    spec, root, _ = lab
    spec["capacity"] = {"dimensions": {"context": [512, 1024]}}

    def mutate_after_trial(event):
        if event["phase"] == "trial_complete":
            assert event["trial"]["eligible"], _diagnostics(event, root / "changed")
            (root / "good.gguf").write_bytes(b"changed after this worker loaded")

    with pytest.raises(ValueError, match="changed after exploration planning"):
        run_exploration(spec, root / "changed", progress=mutate_after_trial)
    saved = json.loads((root / "changed/exploration.json").read_text())
    assert not saved["complete"] and "changed after" in saved["error"]
    assert len((root / "good.gguf.pids").read_text().splitlines()) == 1


def test_preflight_low_memory_never_launches_worker(lab, monkeypatch):
    from llm_optimise import runner

    spec, root, _ = lab
    monkeypatch.setattr(runner, "detect_hardware", lambda: {"ram_available_gib": 0})
    result = run_exploration(spec, root / "no-memory")
    assert result["boundary"]["failure_class"] == "memory_guard"
    assert result["selected_candidate"] is None
    assert not (root / "good.gguf.pids").exists()


def test_empty_kv_contexts_are_not_silently_defaulted(lab):
    spec, _, _ = lab
    with pytest.raises(ValueError, match="contexts"):
        kv_context_preset(spec["experiment"], [], ["f16"])
