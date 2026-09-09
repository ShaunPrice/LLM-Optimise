---
name: llm-optimise
description: Drive an existing LLM-Optimise lab through MCP to measure model quality, speed and memory, prepare datasets, train and evaluate Soup adapters, route chat or coding tasks, and inspect bounded experiment or Docker results.
---

Use the connected LLM-Optimise MCP tools to operate the user's existing lab session. The MCP process shares the GUI's jobs and model ownership. It does not imply that a model, training runtime, Docker engine or hosted connector is installed or working.

1. Use `hardware` and `read_action` with `status` to establish the actual host, models, datasets and active jobs. Discover tool names from the current client when it namespaces MCP tools.
2. Use `discover` with task keywords, then the exact `action_id`, to obtain the current input schema. Send read actions to `read_action` and mutations to `execute_action`. Keep the user's selected model, placement and existing authorization; use the configured provider budget explicitly. Do not retry an uncertain mutation before checking jobs/artifacts.
3. For experiments, use a representative specialised dataset and fixed quality gate. Reuse identical data across comparisons. Keep training, tuning and final held-out evaluation separate. Ask for missing essential constraints only when they cannot be inferred from the request or session.
4. For asynchronous work, retain the returned GUI job ID. Poll `job_status` at sensible intervals, inspect `job_result`, and page through large results using returned offsets. Use `cancel_job` when requested or when the authorized work should stop; cancellation can leave an external call in flight.
5. Interpret the evidence: job completion is distinct from passing acceptance gates. Inspect quality, per-case regressions, sample counts, latency, memory scope, truncation, costs and rejection reasons. Missing GPU/RAM/draft-acceptance telemetry is unknown, never zero. Keep smoke checks distinct from domain validation.

Runtime execution requires operator-configured `--runtime` permissions. Input data stays in the workspace or explicit read roots; artifact reads cover bounded text reports in `runs/` and `validation/`. Do not attempt to bypass these restrictions or fetch credentials. Model catalogue environment fields name keys; their values belong in the App process environment.

For code, inspect the proposed diff before applying within the user's authorized scope. Source/context stale guards remain in force. Generated code executes only through explicit bounded Docker/component/feedback operations. Set `reviewed_execution`, `data_authorized` and `allow_network` only when the user's existing instructions authorize the corresponding execution, provider data transfer or training downloads; missing essential authorization needs clarification. Do not ask again when it is already authorized. Rust repair requires deterministic acceptance cases; component expected labels are scored on the host. A successful training job is followed by a fresh reload and controlled held-out evaluation before claiming improvement.

The `llm-optimise://guide` resource provides a compact transport-independent reminder. Installation, client configuration and OAuth deployment instructions live in the repository's `docs/mcp.md`; hosted ChatGPT/Claude connectivity must be verified in the chosen client, not inferred from a successful local protocol test.
