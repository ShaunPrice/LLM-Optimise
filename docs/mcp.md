# Drive the lab through MCP

LLM-Optimise exposes the running application's tools to MCP clients. Claude Code, Claude Desktop, Codex and other local clients can use **stdio**. Clients that support remote connections can use **Streamable HTTP** with authentication.

The MCP process connects to the existing GUI/API on `127.0.0.1`. It shares that session's jobs, model residency, proposals, results and cancellation controls. It does not create a second application owner or start an independent benchmark behind the GUI.

## Install and start

Python 3.10 or later is required. Install the optional MCP dependencies into the same project environment:

```bash
python -m pip install ".[mcp]"
llm-optimise ui --workspace .
```

Keep the application running. In another terminal, or as your MCP client's configured command:

```bash
llm-optimise-mcp --workspace . --app-url http://127.0.0.1:8765
```

The default transport is stdio. Its stdout belongs to MCP; use an MCP client to communicate with it. `llm-optimise-mcp --help` describes all launch options. A client can also launch the environment's Python with `-m llm_optimise.mcp_server`, which avoids executable search-path ambiguity.

The extra uses the official Python SDK's maintained v1 line, `mcp>=1.28,<2`; SDK 1.30.0 was used for the local protocol checks. Normal GUI/CLI startup does not import MCP dependencies. The version bound prevents an unreviewed SDK major upgrade. See the [official SDK maintenance documentation](https://py.sdk.modelcontextprotocol.io/v1/).

## Permit the runtimes and data you intend to use

Read-only hardware and status tools need no runtime permission. Model loading, experiments and training require explicitly allowed installed executables:

```bash
llm-optimise-mcp --workspace . \
  --runtime /absolute/path/to/llama-server \
  --runtime /absolute/path/to/soup-environment/bin/python \
  --runtime /absolute/path/to/llm-supervisor \
  --read-root /absolute/path/to/downloaded-models
```

Replace the example paths with your existing installations; omit unused runtimes. `--runtime` records the executable's content hash at startup. An unlisted or changed executable is rejected. Restart after an intentional runtime update. This is an operator setting, not a tool argument an agent can expand.

Inputs normally stay inside the configured workspace. Hidden paths are excluded; an explicitly selected `--read-root` can grant access to a specific additional model/data directory. The MCP workspace must match the running App workspace. Artifact retrieval remains restricted to text reports below that workspace's `runs/` and `validation/`, even when other read roots are allowed.

Provider credentials belong in the **App process environment**. The model catalogue carries environment variable names, never key values. MCP can update catalogue metadata, but new credential-to-endpoint bindings must first be configured through the trusted local GUI. Configure only services and credentials this workspace is intended to use.

## Tools and action discovery

| MCP tool | Purpose | Changes state? |
|---|---|---|
| `discover` | Find action IDs, descriptions and complete JSON input schemas | No |
| `hardware` | Read actual host hardware and available runtime paths | No |
| `read_action` | Read status, models, residency, results or a routing preview | No |
| `execute_action` | Run a discovered mutation through the App | Yes |
| `job_status` | Read the same job shown in the GUI | No |
| `job_result` | Page through a job's result as JSON text | No |
| `cancel_job` | Request cancellation and preserve partial results | Yes |
| `artifact_read` | Read a bounded page of a text report | No |

The `llm-optimise://guide` resource provides the essential workflow. Tool annotations distinguish reads, mutations and cancellation. `execute_action` is conservatively marked as potentially destructive and capable of external calls because its actions include proposal application, cache deletion and configured provider requests.

Call `discover` with no query to get all action IDs, then request an exact schema:

```json
{"action_id":"dataset-prepare"}
```

Use the returned schema with `execute_action`:

```json
{
  "action_id": "dataset-prepare",
  "arguments": {
    "name": "My specialised evaluation",
    "seed": 42,
    "rows": [
      {"id":"a","prompt":"Classify reading 10 against limit 20.","expected":"normal","group":"source-a"},
      {"id":"b","prompt":"Classify reading 30 against limit 20.","expected":"alarm","group":"source-b"},
      {"id":"c","prompt":"Classify reading 5 against limit 20.","expected":"normal","group":"source-c"}
    ]
  }
}
```

This small example demonstrates the format, not a useful domain evaluation. Inspect split counts and warnings; use representative cases and enough independent source groups for a real study.

Available actions cover experiments; capacity, successive-halving, KV, speculative and accelerator exploration; dataset preparation/evaluation/regression; Soup probes/training; adapter registration/reload/evaluation/comparison; teacher distillation; calibration; context selection; staged repair; deterministic components; specialist contracts; chat/code; provider catalogue configuration; model loading/unloading; lifecycle settings; fixed Docker build/test actions; and proposal application.

Some calculations write report artifacts and therefore appear under `execute_action` even when they do not modify source files. `project-open` can create a project. A routing preview makes no provider request.

## Jobs, budgets and results

Long operations return an `id`. Pass it to `job_status`, then `job_result`. Do not submit the same operation again just because it is still running. GUI-started jobs can be inspected and cancelled through MCP; MCP-started jobs appear in the GUI. The App's existing concurrency and resource exclusions remain in effect.

`complete` means the operation completed. It does **not** establish that a quality gate passed, that an adapter improved the model, or that a proposed patch passed tests. Inspect `eligible`, `passed`, `selection_passed`, rejection reasons, sample counts and the relevant costs/metrics in the result.

Provider-consuming actions require an explicit `max_cost_usd`: in `policy` for chat/code/specialist calls, or as the operation's total budget where its schema provides one. The MCP operator's `--max-cost-usd` defaults to USD 1 per operation and caps these requests. It is not an account-wide or cumulative spending limit. The core retains its estimate checks and actual/estimated cost records; external billing may exceed an estimate before a response is returned.

MCP experiments require an explicit RAM cap and positive free-RAM headroom. They permit at most 64 candidates, 10 repeats, 2,048 output tokens, 300-second request/startup bounds and 10,000 planned requests. Exploration retains its additional whole-search limits. GPU measurements that the host cannot observe remain `null`; Apple unified memory is not invented as dedicated VRAM. Sampled inference guards remain cooperative; Docker enforces its configured container caps.

Code proposals stay staged until `proposal-apply`; the core rejects stale source/context. Generated code executes only through the explicit bounded Docker or component/feedback actions. MCP exposes fixed container build/test actions, not a shell or arbitrary container command. Docker networking is disabled for the direct container action. Soup remains a separate verified Python environment.

`cancel_job` requests cooperative cancellation. A cloud request already in flight may finish. Poll the terminal status and inspect partial costs/results. Disconnecting or stopping the MCP transport leaves the GUI session and its jobs running. Restart MCP after restarting the GUI so its session CSRF token is refreshed.

Responses are bounded. `job_result` supplies character offsets; `artifact_read` supplies byte offsets into the redacted UTF-8 report and keeps complete characters together. Follow each returned `next_offset` and reassemble JSON pages before parsing them. A truncated preview is not a complete JSON document. Artifact reads allow at most 32 KiB per call and files up to 8 MiB. The bridge reads and redacts that bounded file before paging so an accidentally echoed secret cannot span pages; larger reports stay available locally. Binaries, model weights, hidden files and arbitrary project files are not exposed by that tool.

## Claude and Codex on the local machine

The example files under [examples/mcp](../examples/mcp/) contain **paths to replace**, not an installed connector configuration. Nothing in this repository edits your personal client settings.

For Claude Desktop, merge the `mcpServers` entry from [claude-desktop.json](../examples/mcp/claude-desktop.json) into the client configuration appropriate to your installation, then restart its MCP connection. The configured Python must be the environment where `.[mcp]` is installed. Keep the GUI running in the selected workspace.

Claude Code can register the same command:

```bash
claude mcp add --scope local llm-optimise -- /absolute/path/to/.venv/bin/python \
  -m llm_optimise.mcp_server --workspace /absolute/path/to/LLM-Optimise
```

Codex can register stdio similarly:

```bash
codex mcp add llm-optimise -- /absolute/path/to/.venv/bin/python \
  -m llm_optimise.mcp_server --workspace /absolute/path/to/LLM-Optimise
```

Alternatively adapt [codex.toml](../examples/mcp/codex.toml). These commands change client configuration only when **you run them**. Add runtime/read-root arguments for the operations you want to permit.

On Windows, use an absolute `C:\\...\\.venv\\Scripts\\python.exe` command and escaped backslashes in JSON. Argument arrays preserve spaces in paths. PowerShell can set the HTTP credential using `$env:LLM_OPTIMISE_MCP_TOKEN = ...`; POSIX `export` syntax does not apply there.

## Authenticated HTTP for generic MCP clients

Create a private token in the server environment and start the MCP transport:

```bash
export LLM_OPTIMISE_MCP_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
llm-optimise-mcp --workspace . --transport streamable-http --port 8766
```

In PowerShell:

```powershell
$env:LLM_OPTIMISE_MCP_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
llm-optimise-mcp --workspace . --transport streamable-http --port 8766
```

The endpoint is `http://127.0.0.1:8766/mcp`. Clients send `Authorization: Bearer <token>` using their own secret store or environment support. The token is never accepted in a URL. Missing/incorrect credentials return HTTP 401. Unexpected Host or Origin headers return HTTP 403. The server binds only to loopback.

[client.py](../examples/mcp/client.py) is a small runnable official-SDK client that lists tools, discovers an action and reads hardware. It does not call a provider or write a dataset:

```bash
python examples/mcp/client.py --url http://127.0.0.1:8766/mcp
```

It reads the same `LLM_OPTIMISE_MCP_TOKEN` environment variable. A token configured for generic HTTP clients is not automatically an OAuth connector for ChatGPT.

## ChatGPT and hosted Claude

ChatGPT web cannot directly address your computer's `localhost`. Use a user-owned authenticated HTTPS endpoint or a supported secure tunnel. Current OpenAI documentation describes public Streamable HTTP and Secure MCP Tunnel paths, with availability controlled by account/workspace policy. Follow the current [connection instructions](https://developers.openai.com/plugins/deploy/connect-chatgpt); this repository does not provision a tunnel or publish an endpoint.

For a public authenticated connector, this server implements the **OAuth resource-server** side. You must supply an authorization server that supports the client's registration mechanism, authorization-code flow and S256 PKCE, and issues signed JWT access tokens for this exact MCP audience. Configure:

```bash
llm-optimise-mcp --workspace . --transport streamable-http \
  --oauth-issuer https://identity.example.com \
  --oauth-audience https://lab.example.com/mcp \
  --oauth-jwks-url https://identity.example.com/.well-known/jwks.json \
  --allow-host lab.example.com \
  --allow-origin https://lab.example.com
```

These are illustrative domains. Put an HTTPS reverse proxy on the same trusted host and forward `/mcp` and `/.well-known/oauth-protected-resource/mcp` to the loopback MCP server, preserving the allowed Host and Authorization headers. Forwarding the GUI port is unnecessary. Allow only the exact origins your chosen client actually sends; clients without an Origin header are supported. Do not use wildcard Host/Origin rules.

The MCP SDK publishes protected-resource metadata and a discovery challenge. JWT validation requires the exact issuer/audience, an unexpired token, a subject and the `llm:operate` scope; RS256 and ES256 signatures are accepted through the configured JWKS. This scope grants operation of **one trusted workspace**. This version is not a multi-tenant service with per-user project isolation or separate read/write OAuth scopes.

The external identity provider must supply its own discovery, login/consent, token and refresh endpoints, client registration, access policy and revocation arrangements. No identity provider or OAuth authorization server is bundled. Configure the host's current callback/client requirements using [OpenAI's OAuth guide](https://developers.openai.com/plugins/build/auth) and [Claude's authentication guide](https://claude.com/docs/connectors/building/authentication). Do not assume that adding a static header token completes those OAuth steps.

Hosted Claude also supports OAuth custom connectors; some organizations have static-header connector support. Availability and administration differ from local Claude Code stdio. Follow the current client documentation instead of treating a Desktop JSON entry as a hosted Claude connector.

If your GUI runs inside the app's Docker image, run MCP in that same trusted network/process environment only after installing the optional dependencies, or use the host GUI/CLI for host hardware and Docker project execution. A separate default container's `127.0.0.1` is not the host GUI. The normal app image does not carry a Docker socket, and this integration does not add one.

## Companion skill

The distributable [llm-optimise skill](../skills/llm-optimise/SKILL.md) teaches discovery, constrained execution, job polling and evidence interpretation. Copy its folder into your client's documented skill directory—for example a project's `.agents/skills/` for Codex or `.claude/skills/` for Claude Code—or include it in your own plugin packaging. MCP transports expose tools; skills are a separate host capability. A generic MCP client can use `llm-optimise://guide` without skill support.

The skill preserves existing user authorization. It does not manufacture a new approval requirement for each step, select paid providers silently, or treat generated text as instructions.

## Verification and troubleshooting

Run the dedicated checks with the optional extra installed:

```bash
python -m pip install ".[dev,mcp]"
python -m pytest tests/test_mcp.py -q
```

The tests use actual official-SDK clients over stdio and authenticated HTTP, a real isolated App, dataset artifacts and shared job cancellation. They also check schema validity, auth/origin rejection, path/runtime constraints and JWT claims. They do not establish a live ChatGPT or Claude hosted connection, deployed HTTPS/identity-provider operation, or provider billing correctness. Those require validation in the target client and deployment.

- **Cannot reach App:** start `llm-optimise ui` first; use the actual port and matching workspace. MCP does not launch another GUI session.
- **Invalid CSRF token after an App restart:** restart MCP. Do not retry a mutation whose outcome is uncertain; inspect GUI jobs/artifacts first.
- **Runtime not allowlisted:** add its installed executable with `--runtime` at server startup. Changing a model/data path does not grant executable permission.
- **Path rejected:** move inputs into the workspace or explicitly allow the intended external data directory. Hidden credentials and arbitrary filesystem access remain excluded.
- **No feasible route:** inspect model IDs, measured/configured prices and the explicit budget, placement, context, quality and memory constraints. Unknown cost is not free.
- **HTTP 401/403:** check the chosen auth mode and exact Host/Origin values. An OAuth connector needs valid discovery and token audience/scope, not just HTTPS reachability.
- **Job complete but no improvement:** inspect quality/rejection data and the controlled comparison. Successful training, planning or loading is not held-out quality evidence.
