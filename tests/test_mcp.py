"""Real stdio and HTTP MCP clients against one real, isolated GUI App session."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from llm_optimise.mcp_bridge import AppBridge
from llm_optimise.mcp_catalog import CATALOG
from llm_optimise.mcp_server import jwt_verifier
from llm_optimise.server import make_server


@pytest.fixture
def app(tmp_path):
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples/tasks.jsonl").write_text(
        json.dumps({"id": "x", "prompt": "say ok", "expected": "ok"}) + "\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models/fixture.gguf").write_bytes(b"planning fixture; no inference")
    yield server, tmp_path, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def command(app):
    _, root, url = app
    return [
        sys.executable,
        "-m",
        "llm_optimise.mcp_server",
        "--workspace",
        str(root),
        "--app-url",
        url,
        "--runtime",
        sys.executable,
    ]


def env():
    return {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}


def data(result):
    assert not result.isError, result.content
    return result.structuredContent or json.loads(result.content[0].text)


async def exercise(session, app):
    server, root, _ = app
    await session.initialize()
    tools = (await session.list_tools()).tools
    by_name = {tool.name: tool for tool in tools}
    assert by_name["read_action"].annotations.readOnlyHint is True
    assert by_name["execute_action"].annotations.readOnlyHint is False
    assert by_name["cancel_job"].annotations.readOnlyHint is False
    discovered = data(await session.call_tool("discover", {"action_id": "dataset-prepare"}))
    assert discovered["actions"][0]["input_schema"]["properties"]["rows"]["type"] == "array"
    assert (await session.read_resource("llm-optimise://guide")).contents
    assert "ram_total_gib" in data(await session.call_tool("hardware", {}))
    assert data(await session.call_tool("read_action", {"action_id": "models"}))["models"] == []
    wrong = await session.call_tool("read_action", {"action_id": "cache-clear"})
    assert wrong.isError
    prepared = data(
        await session.call_tool(
            "execute_action",
            {
                "action_id": "dataset-prepare",
                "arguments": {
                    "name": "MCP fixture",
                    "rows": [
                        {"id": str(i), "prompt": f"case {i}", "expected": "ok", "group": str(i)}
                        for i in range(10)
                    ],
                },
            },
        )
    )
    artifact = Path(prepared["artifact_dir"]) / "manifest.json"
    assert artifact.is_file()
    read = data(
        await session.call_tool(
            "artifact_read", {"path": str(artifact.relative_to(root)), "max_bytes": 128}
        )
    )
    assert read["next_offset"] == 128 and not read["eof"]
    assert (await session.call_tool("artifact_read", {"path": "../outside.txt"})).isError
    plan = {
        "mode": "capacity",
        "experiment": {
            "name": "MCP planning fixture",
            "dataset": "examples/tasks.jsonl",
            "max_tokens": 16,
            "repeats": 1,
            "warmup": 0,
            "limits": {"max_rss_gib": 1, "min_available_gib": 0.001},
            "candidates": [
                {
                    "name": "fixture",
                    "model": "models/fixture.gguf",
                    "executable": sys.executable,
                    "context": 512,
                }
            ],
        },
        "capacity": {"dimensions": {"context": [512]}},
    }
    submitted = data(
        await session.call_tool("execute_action", {"action_id": "explore-plan", "arguments": plan})
    )
    assert submitted["id"] in server.app.jobs
    for _ in range(100):
        status = data(await session.call_tool("job_status", {"job_id": submitted["id"]}))
        if status["status"] != "running":
            break
        await asyncio.sleep(0.02)
    assert status["status"] == "complete", status
    result = data(
        await session.call_tool("job_result", {"job_id": submitted["id"], "max_characters": 32000})
    )
    assert result["eof"] and json.loads(result["text"])["mode"] == "capacity"
    # Work started by the GUI shares the same ID and cancellation event through MCP.
    started = threading.Event()

    def gui_work(job):
        started.set()
        job["cancel"].wait(5)
        return {"cancelled": job["cancel"].is_set()}

    gui_job = server.app.start_job("fixture", gui_work)["id"]
    assert started.wait(1)
    cancelled = data(await session.call_tool("cancel_job", {"job_id": gui_job}))
    assert cancelled["cancellation_requested"]
    assert server.app.jobs[gui_job]["cancel"].is_set()
    for _ in range(100):
        status = data(await session.call_tool("job_status", {"job_id": gui_job}))
        if status["status"] != "running":
            break
        await asyncio.sleep(0.02)
    assert status["status"] == "cancelled"
    from llm_optimise.workspace import prepare_proposal

    server.app.project("source-guard")
    project = root / "projects/source-guard"
    (project / "main.py").write_text("original")
    proposal = prepare_proposal(
        project,
        json.dumps({"summary": "fixture", "files": [{"path": "main.py", "content": "proposed"}]}),
        {"main.py": "original"},
    )
    proposal_id = "a" * 32
    server.app.proposals[proposal_id] = (project, proposal)
    (project / "main.py").write_text("changed independently")
    rejected = await session.call_tool(
        "execute_action", {"action_id": "proposal-apply", "arguments": {"proposal_id": proposal_id}}
    )
    assert rejected.isError
    assert (project / "main.py").read_text() == "changed independently"


def test_real_stdio_client_and_shared_app_jobs(app):
    async def run():
        args = command(app)
        async with (
            stdio_client(StdioServerParameters(command=args[0], args=args[1:], env=env())) as (
                read,
                write,
            ),
            ClientSession(read, write) as session,
        ):
            await exercise(session, app)

    asyncio.run(run())


def test_real_authenticated_http_client_origin_and_auth(app):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = "mcp-test-token-" + "a" * 40
    process = subprocess.Popen(
        [*command(app), "--transport", "streamable-http", "--port", str(port)],
        env={**env(), "LLM_OPTIMISE_MCP_TOKEN": token},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail(process.stderr.read().decode())
            try:
                unauth = httpx.get(url, timeout=0.2, trust_env=False)
                break
            except httpx.TransportError:
                time.sleep(0.05)
        else:
            pytest.fail("HTTP MCP did not start within 5 seconds")
        assert unauth.status_code == 401
        assert (
            httpx.get(url, headers={"Authorization": "Bearer wrong"}, trust_env=False).status_code
            == 401
        )
        assert (
            httpx.get(
                url,
                headers={"Authorization": "Bearer " + token, "Origin": "https://evil.example"},
                trust_env=False,
            ).status_code
            == 403
        )
        assert (
            httpx.get(
                url,
                headers={"Authorization": "Bearer " + token, "Host": "evil.example"},
                trust_env=False,
            ).status_code
            == 403
        )

        sample = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "examples/mcp/client.py"),
                "--url",
                url,
            ],
            env={**env(), "LLM_OPTIMISE_MCP_TOKEN": token},
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert sample.returncode == 0, sample.stderr
        assert '"hardware"' in sample.stdout

        async def run():
            async with (
                httpx.AsyncClient(
                    headers={"Authorization": "Bearer " + token}, trust_env=False
                ) as http_client,
                streamable_http_client(url, http_client=http_client) as (
                    read,
                    write,
                    _,
                ),
                ClientSession(read, write) as session,
            ):
                await exercise(session, app)

        asyncio.run(run())
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def test_action_schemas_are_valid_and_cover_workbench():
    from jsonschema import Draft202012Validator

    from llm_optimise.workbench import OPERATIONS

    for item in CATALOG:
        Draft202012Validator.check_schema(item["input_schema"])
    assert set(OPERATIONS) - {item["id"] for item in CATALOG} == set()


def test_bridge_refuses_wrong_workspace_and_nonloopback(app, tmp_path):
    _, root, url = app
    with pytest.raises(ValueError, match="workspace differs"):
        AppBridge(root / "other", url).state()
    for url in (
        "http://example.com:8765",
        "http://localhost:8765",
        "http://127.0.0.1:8765/path",
        "http://x:password@127.0.0.1:8765",
    ):
        with pytest.raises(ValueError, match="app-url"):
            AppBridge(root, url)


def test_paths_runtime_and_raw_credentials_stay_bounded(app):
    from jsonschema import ValidationError

    _, root, url = app
    bridge = AppBridge(root, url, runtimes=[sys.executable])
    (root / ".env").write_text("SECRET=do-not-read")
    for path in (str(root.parent), ".env"):
        with pytest.raises(ValueError, match="allowed data roots"):
            bridge.read_path(path)
    with pytest.raises(ValueError, match="operator-allowlisted"):
        bridge.runtime(str(root / "models/fixture.gguf"))
    with pytest.raises(ValidationError):
        bridge.call(
            "container",
            {"project": "safe", "runtime": "python", "action": "test", "command": ["sh"]},
            read=False,
        )
    assert not (root / "projects/safe").exists()
    assert bridge.redact({"token": "private", "api_key_env": "MY_KEY"}) == {
        "token": "<redacted>",
        "api_key_env": "MY_KEY",
    }


def test_symlink_artifact_escape_is_rejected(app, tmp_path):
    _, root, url = app
    external = root.parent / (root.name + "-secret.txt")
    external.write_text("external data")
    (root / "runs").mkdir(exist_ok=True)
    try:
        (root / "runs/escape.txt").symlink_to(external)
    except OSError:
        pytest.skip("host denies symlink creation")
    with pytest.raises(ValueError):
        AppBridge(root, url).artifact("runs/escape.txt")


def test_artifact_pages_preserve_unicode_and_redact_across_boundaries(app, monkeypatch):
    _, root, url = app
    secret = "private-key-spanning-many-pages"
    monkeypatch.setenv("MCP_TEST_SECRET", secret)
    (root / "runs").mkdir(exist_ok=True)
    (root / "runs/unicode.json").write_text(
        json.dumps({"text": "é🧪" + secret}, ensure_ascii=False), encoding="utf-8"
    )
    bridge = AppBridge(root, url)
    pieces, offset = [], 0
    while True:
        page = bridge.artifact("runs/unicode.json", offset, 4)
        assert len(page["text"].encode("utf-8")) <= 4
        pieces.append(page["text"])
        offset = page["next_offset"]
        if page["eof"]:
            break
    result = "".join(pieces)
    assert json.loads(result) == {"text": "é🧪<redacted>"}
    assert secret not in result
    assert page["total_bytes"] == len(result.encode("utf-8"))
    with pytest.raises(ValueError, match="UTF-8 boundary"):
        bridge.artifact("runs/unicode.json", result.encode("utf-8").index(b"\xc3") + 1)
    with (root / "runs/large.log").open("wb") as stream:
        stream.truncate(8 * 1024**2 + 1)
    with pytest.raises(ValueError, match="exceeds 8 MiB"):
        bridge.artifact("runs/large.log")


def test_runtime_allowlist_does_not_accept_another_venv_spelling(app):
    _, root, url = app
    executable = root / "python"
    executable.write_bytes(b"shared interpreter fixture")
    approved, other = root / "approved-python", root / "other-python"
    try:
        approved.symlink_to(executable)
        other.symlink_to(executable)
    except OSError:
        pytest.skip("host denies symlink creation")
    bridge = AppBridge(root, url, runtimes=[str(approved)])
    assert bridge.runtime(str(approved)) == str(approved)
    for spelling in (str(other), str(executable)):
        with pytest.raises(ValueError, match="operator-allowlisted"):
            bridge.runtime(spelling)


def test_http_security_gate_rejects_non_ascii_headers_cleanly():
    from llm_optimise.mcp_server import SecurityGate

    async def unreachable(*_args):
        raise AssertionError("unauthenticated request reached the application")

    gate = SecurityGate(unreachable, ["localhost:8766"], [], token="t" * 32)

    async def check(headers, expected):
        sent = []

        async def send(message):
            sent.append(message)

        await gate(
            {"type": "http", "method": "GET", "path": "/mcp", "headers": headers},
            None,
            send,
        )
        assert sent[0]["status"] == expected

    async def run():
        host = [(b"host", b"localhost:8766")]
        await check(host + [(b"authorization", b"Bearer \xff")], 401)
        await check(host + [(b"origin", b"\xff")], 403)
        await check(host + [(b"host", b"localhost:8766")], 400)

    asyncio.run(run())


def test_jwt_verifier_validates_issuer_audience_scope_and_expiry(monkeypatch):
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(
        jwt.PyJWKClient,
        "get_signing_key_from_jwt",
        lambda *_: type("Key", (), {"key": key.public_key()})(),
    )
    verifier = jwt_verifier(
        "https://issuer.example", "https://mcp.example/mcp", "https://issuer.example/keys"
    )
    claims = {
        "iss": "https://issuer.example",
        "aud": "https://mcp.example/mcp",
        "sub": "operator",
        "exp": int(time.time()) + 60,
        "scope": "llm:operate",
    }

    async def check():
        assert await verifier.verify_token(jwt.encode(claims, key, algorithm="RS256"))
        for patch in (
            {"iss": "https://wrong.example"},
            {"aud": "https://wrong.example"},
            {"scope": "other"},
            {"exp": 1},
        ):
            assert (
                await verifier.verify_token(jwt.encode(claims | patch, key, algorithm="RS256"))
                is None
            )

    asyncio.run(check())


def test_http_refuses_start_without_auth(app):
    values = env()
    values.pop("LLM_OPTIMISE_MCP_TOKEN", None)
    reply = subprocess.run(
        [*command(app), "--transport", "streamable-http"],
        env=values,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert reply.returncode == 2 and "requires a bearer token" in reply.stderr


def test_oauth_resource_discovery_and_http_challenge(app):
    from mcp.server.auth.settings import AuthSettings

    from llm_optimise.mcp_server import SecurityGate, create_server

    _, root, app_url = app
    audience = "https://mcp.example/mcp"
    verifier = jwt_verifier("https://issuer.example", audience, "https://issuer.example/jwks")
    auth = AuthSettings(
        issuer_url="https://issuer.example",
        resource_server_url=audience,
        required_scopes=["llm:operate"],
        validate_token_resource=True,
    )
    server = create_server(
        AppBridge(root, app_url),
        hosts=["mcp.example"],
        origins=["https://mcp.example"],
        auth=auth,
        verifier=verifier,
    )
    wrapped = SecurityGate(
        server.streamable_http_app(),
        ["mcp.example"],
        ["https://mcp.example"],
        oauth_metadata={
            "resource": audience,
            "authorization_servers": ["https://issuer.example"],
            "scopes_supported": ["llm:operate"],
        },
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=wrapped), base_url="https://mcp.example"
        ) as client:
            metadata = await client.get("/.well-known/oauth-protected-resource/mcp")
            assert metadata.status_code == 200, metadata.text
            assert metadata.json()["resource"] == audience
            assert metadata.json()["authorization_servers"] == ["https://issuer.example"]
            denied = await client.post("/mcp", json={})
            assert denied.status_code == 401
            assert "resource_metadata=" in denied.headers["www-authenticate"]
            assert (
                await client.get(
                    "/.well-known/oauth-protected-resource/mcp",
                    headers={"Origin": "https://evil.example"},
                )
            ).status_code == 403

    asyncio.run(run())


def test_provider_budget_and_runtime_change_precede_app_mutation(app):
    from jsonschema import ValidationError

    _, root, url = app
    runtime = root / "runtime"
    runtime.write_bytes(b"trusted installed runtime fixture")
    bridge = AppBridge(root, url, runtimes=[str(runtime)], max_cost_usd=0.1)
    runtime.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed"):
        bridge.runtime(str(runtime))
    for budget in (None, 0.2, -1):
        with pytest.raises((ValueError, ValidationError)) as caught:
            bridge.call(
                "chat", {"message": "hello", "policy": {"max_cost_usd": budget}}, read=False
            )
        assert "max_cost_usd" in str(caught.value)
    assert app[0].app.jobs == {}
