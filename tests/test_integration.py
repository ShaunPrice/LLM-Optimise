import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import psutil
import pytest

from llm_optimise import backend as backend_module
from llm_optimise.agent import run_agent
from llm_optimise.backend import Backend
from llm_optimise.config import Candidate, Experiment, Limits
from llm_optimise.providers import complete
from llm_optimise.quality import Task
from llm_optimise.routing import ProviderModel, RoutePolicy, choose_route
from llm_optimise.runner import run_experiment
from llm_optimise.server import make_server


@pytest.fixture
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        requests = []
        redirect = False
        truncated = False

        def log_message(self, *_):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            Handler.requests.append((self.path, data, dict(self.headers)))
            if Handler.redirect:
                self.send_response(302)
                self.send_header("Location", "https://example.com")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/apply-template":
                self.wfile.write(json.dumps({"prompt": "hello"}).encode())
            elif self.path == "/completion":
                events = [
                    {"content": "yes", "stop": False},
                    {
                        "content": "",
                        "stop": True,
                        "stop_type": "limit" if Handler.truncated else "eos",
                        "timings": {
                            "predicted_n": 1,
                            "predicted_per_second": 20,
                            "prompt_n": 1,
                            "prompt_per_second": 30,
                        },
                    },
                ]
                for e in events:
                    self.wfile.write(("data: " + json.dumps(e) + "\n\n").encode())
            elif self.path == "/messages":
                self.wfile.write(
                    json.dumps(
                        {
                            "content": [{"type": "text", "text": "hello"}],
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 5, "output_tokens": 2},
                        }
                    ).encode()
                )
            else:
                self.wfile.write(
                    json.dumps(
                        {
                            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                        }
                    ).encode()
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", Handler
    server.shutdown()
    server.server_close()
    thread.join()


def model(url, provider="openai", **kw):
    return ProviderModel(
        id="test",
        model="test-model",
        provider=provider,
        location="local",
        base_url=url,
        context_window=10000,
        max_output_tokens=2000,
        input_cost_per_million=0,
        output_cost_per_million=0,
        **kw,
    )


def test_local_provider_protocol_and_routing(endpoint):
    url, handler = endpoint
    result = run_agent([model(url)], [{"role": "user", "content": "Hi"}], max_tokens=10)
    assert result["completion"]["text"] == "hello"
    assert handler.requests[0][1]["max_completion_tokens"] == 10
    assert len(handler.requests) == 1
    assert result["completion"]["accounted_cost_usd"] == 0
    assert result["route"]["model_id"] == "test"


def test_anthropic_adapter_request(endpoint, monkeypatch):
    url, handler = endpoint
    monkeypatch.setenv("TEST_LLM_KEY", "secret-not-for-output")
    result = complete(
        model(url, provider="anthropic", api_key_env="TEST_LLM_KEY"),
        [{"role": "system", "content": "Helpful"}, {"role": "user", "content": "Hi"}],
        10,
    )
    assert result["text"] == "hello"
    assert handler.requests[0][1]["system"] == "Helpful"
    assert handler.requests[0][2]["X-Api-Key"] == "secret-not-for-output"
    assert "secret-not-for-output" not in json.dumps(result)


def test_provider_redirect_refused(endpoint):
    url, handler = endpoint
    handler.redirect = True
    with pytest.raises(RuntimeError, match="HTTP 302"):
        complete(model(url), [{"role": "user", "content": "hello"}], 10)
    assert len(handler.requests) == 1


def test_llama_streaming_measures_tokens_and_truncation(endpoint, tmp_path):
    url, handler = endpoint
    b = Backend(Candidate(name="test", model="unused"), tmp_path / "log")
    b.endpoint = url
    got = b.generate(Task("t", "hi", "yes"), 10, 42, 2, lambda: None)
    assert got.text == "yes" and got.output_tokens == 1 and got.decode_tokens_s == 20
    assert got.ttft_s is not None and got.latency_s >= got.ttft_s
    handler.truncated = True
    assert b.generate(Task("t", "hi", "yes"), 10, 42, 2, lambda: None).truncated


def test_partial_stream_cannot_pass_as_complete(tmp_path, monkeypatch):
    b = Backend(Candidate(name="test", model="unused"), tmp_path / "log")
    b.endpoint = "http://127.0.0.1"
    monkeypatch.setattr(backend_module, "request_json", lambda *a, **k: {"prompt": "hello"})
    monkeypatch.setattr(backend_module, "stream_json", lambda *a, **k: iter([{"content": "yes"}]))
    with pytest.raises(RuntimeError, match="completion marker"):
        b.generate(Task("t", "hi", "yes"), 10, 42, 2, lambda: None)


@pytest.mark.parametrize("provider", ["llama.cpp", "openai"])
def test_submillisecond_requests_survive_a_coarse_monotonic_clock(tmp_path, monkeypatch, provider):
    ticks = iter([10.0, 10.0001, 10.0002, 10.0003])
    monkeypatch.setattr(
        backend_module,
        "time",
        SimpleNamespace(monotonic=lambda: 10.0, perf_counter=lambda: next(ticks)),
    )
    monkeypatch.setattr(backend_module, "request_json", lambda *a, **k: {"prompt": "hello"})
    event = (
        {"content": "yes", "stop": True}
        if provider == "llama.cpp"
        else {"choices": [{"delta": {"content": "yes"}, "finish_reason": "stop"}]}
    )
    budgets = []

    def stream(_url, _payload, remaining, _key, _check):
        budgets.append(remaining)
        yield event

    monkeypatch.setattr(backend_module, "stream_json", stream)
    backend = Backend(Candidate(name="timer", model="fixture", backend=provider), tmp_path / "log")
    backend.endpoint = "http://127.0.0.1"
    measured = backend.generate(Task("timer", "hi", "yes"), 10, 42, 2, lambda: None)
    assert measured.latency_s == pytest.approx(0.0003)
    assert measured.ttft_s == pytest.approx(0.0002)
    assert budgets == [pytest.approx(1.9999)]


def test_process_cleanup(tmp_path):
    b = Backend(Candidate(name="test", model="unused"), tmp_path / "log")
    b.process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    pid = b.pid
    b.close()
    assert b.process.poll() is not None
    assert not psutil.pid_exists(pid)


def test_budget_unknown_gpu_rejects_and_failed_trials_preserved(tmp_path, monkeypatch):
    from llm_optimise import runner
    from llm_optimise.backend import Generation
    from llm_optimise.config import file_digest

    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"id": "a", "prompt": "hi", "expected": "yes"}) + "\n")

    class FakeBackend:
        pid = None
        version = "fake"
        command = None
        key = None

        def __init__(self, c, *a):
            self.c = c

        def launch(self):
            if self.c.name == "broken":
                raise RuntimeError("deliberate failure")

        def ready(self, *a):
            pass

        def close(self):
            pass

        def generate(self, *a):
            return Generation("yes", 0.1, 0.02, 1, 20)

    monkeypatch.setattr(runner, "Backend", FakeBackend)
    monkeypatch.setattr(runner, "_provenance", lambda x: {str(data): file_digest(data)})
    exp = Experiment(
        name="integration",
        dataset=str(data),
        warmup=0,
        repeats=1,
        candidates=(
            Candidate("works", "fake", backend="openai", endpoint="http://127.0.0.1/v1"),
            Candidate("broken", "fake", backend="openai", endpoint="http://127.0.0.1/v1"),
        ),
        limits=Limits(min_available_gib=0, max_gpu_gib=2),
    )
    result = run_experiment(exp, tmp_path / "out")
    assert result["frontier"] == [] and not result["complete"]
    assert {t["status"] for t in result["trials"]} == {"failed", "complete"}
    with pytest.raises(ValueError, match="not empty"):
        run_experiment(exp, tmp_path / "out")


def test_http_ui_csrf_and_models(tmp_path):
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        req = Request(
            url + "/api/models", data=b'{"models":[]}', headers={"Content-Type": "application/json"}
        )
        with pytest.raises(HTTPError) as err:
            urlopen(req)
        assert err.value.code == 403
        req.add_header("X-LLM-Token", server.app.token)
        assert json.load(urlopen(req)) == {"models": []}
        req.add_header("Origin", "https://evil.example")
        with pytest.raises(HTTPError) as err:
            urlopen(req)
        assert err.value.code == 403
        state = json.load(urlopen(url + "/api/state"))
        assert "hardware" in state and state["models"] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_router_local_memory_budget_and_public_destination():
    low = model("http://localhost:8000", ram_gib=2, gpu_gib=0)
    assert (
        choose_route(
            [low], RoutePolicy(placement="local", objective="cost", max_ram_gib=3), 100, 100
        )["model_id"]
        == "test"
    )
    with pytest.raises(ValueError, match="ram_gib"):
        choose_route(
            [low], RoutePolicy(placement="local", objective="cost", max_ram_gib=1), 100, 100
        )
    from llm_optimise.network import validate_local_destination

    with pytest.raises(ValueError, match="local placement"):
        validate_local_destination("http://8.8.8.8/v1")


def test_schema_output_only_when_provider_declares_support(endpoint):
    url, handler = endpoint
    run_agent(
        [model(url, supports_json_schema=True)],
        [{"role": "user", "content": "code"}],
        max_tokens=10,
        capability="code",
    )
    schema = handler.requests[-1][1]["response_format"]
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["schema"]["required"] == ["summary", "files"]
    run_agent([model(url)], [{"role": "user", "content": "code"}], max_tokens=10, capability="code")
    assert "response_format" not in handler.requests[-1][1]


def test_openrouter_cost_and_no_gateway_fallback(monkeypatch):
    import io

    from llm_optimise import providers

    captured = []

    def respond(request, **kwargs):
        captured.append(json.loads(request.data))
        return io.BytesIO(
            json.dumps(
                {
                    "id": "generation-test",
                    "model": "openai/test",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.000004},
                }
            ).encode()
        )

    monkeypatch.setattr(providers, "open_request", respond)
    monkeypatch.setenv("TEST_ROUTER_KEY", "fixture-only")
    m = ProviderModel(
        id="or",
        model="openai/test",
        provider="openai",
        location="cloud",
        base_url="https://openrouter.ai/api/v1",
        context_window=10000,
        max_output_tokens=100,
        api_key_env="TEST_ROUTER_KEY",
    )
    result = complete(m, [{"role": "user", "content": "hello"}], 10)
    assert captured[0]["provider"] == {"allow_fallbacks": False, "require_parameters": True}
    assert result["provider_reported_cost_usd"] == 0.000004
    assert result["accounted_cost_usd"] is None
    assert result["response_id"] == "generation-test"
    assert "fixture-only" not in json.dumps(result)


def test_nonbenchmark_validation_artifacts_do_not_break_lab_state(tmp_path):
    from llm_optimise.server import App

    directory = tmp_path / "validation" / "provider"
    directory.mkdir(parents=True)
    (directory / "results.json").write_text(json.dumps({"status": "passed", "calls": 14}))
    assert App(tmp_path)._results() == []
