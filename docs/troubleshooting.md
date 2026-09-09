# Troubleshooting

Start with `llm-optimise doctor`, the error message, the experiment's `results.json`, and the matching runtime log. A successful install, a discovered executable, a responding endpoint and a passed task evaluation establish different things.

## Installation and startup

| Symptom | What to check |
| --- | --- |
| `externally-managed-environment` / PEP 668 | Create a virtual environment and run its Python/pip. Do not force a system-Python installation. See [getting started](getting-started.md). |
| `llm-optimise: command not found` | Activate the environment or invoke its executable directly. Confirm you installed this repository into that same environment. |
| PowerShell refuses activation | Use `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\llm-optimise.exe` directly; activation is optional. |
| Import or Python syntax error | Confirm Python is at least 3.10 and the shell is using the intended environment. Keep Soup in its separate compatible environment. |
| Port already in use | Run `llm-optimise ui --workspace . --port 8766 --open`. |
| Browser shows a stale page or mutation fails | Reload from the currently running local server. Its request token changes after restart. |
| `untrusted request origin or Host` | Use the printed `http://127.0.0.1:PORT` or `http://localhost:PORT` URL. Public hostnames, LAN URLs and reverse-proxy access are not the supported local interface. |
| Results or models seem missing | Check the workspace shown in the GUI. A different `--workspace` has a different catalogue, projects and discovered runs. |

## Model and runtime loading

**Executable unavailable.** Put a compatible `llama-server` on `PATH` or set its exact executable path. Windows JSON paths can use forward slashes (`C:/models/model.gguf`) to avoid accidental backslash escapes. Quote shell paths containing spaces.

**Model file missing.** The lab does not download the sample model. CLI relative paths follow the experiment JSON's directory; GUI relative paths follow the selected workspace. Inspect the resolved candidate path in validation/configuration and use an existing compatible GGUF file.

**GPU requested but not used.** `doctor` detects selected hardware/runtime information, not accelerator build correctness. Check your runtime's build and driver support, then inspect `trial-*.log` for actual offload. Start with `gpu_layers: 0` to isolate loading from GPU support. `-1` requests all supported layers but does not prove acceleration. Unsupported Apple GPU telemetry can remain unavailable even when Metal execution works.

**Server exits during startup.** Open its log for incompatible model architecture, unknown flags, memory allocation failure or accelerator errors. Check your `llama-server` version and supported flags. Changing the executable after a run changes provenance, so use a fresh output directory.

**Startup timeout.** Confirm the model is not failing in the log. Large model loading, slow storage and a cold filesystem cache can take longer than the default 120 seconds. Raise `startup_timeout_s` only after identifying a plausible slow load, and keep load-time comparisons separate from warmed request latency.

**Quantised V-cache error.** Use `flash_attention: "on"` with non-`f16` `cache_type_v`, and verify the runtime supports the combination. Return to `f16` for an initial compatibility test.

## Experiment completion and quality

**No frontier / exit 2.** Inspect each trial's `status` and `rejection_reasons`. A completed low-quality model is correctly excluded. Missing required memory/TTFT data also prevents eligibility. Fix the failing condition or deliberately revise the task requirement; do not relabel a failed gate as an optimisation success.

**Insufficient available RAM.** Close competing workloads, reduce model/context/batch sizes, or use a more suitable runtime configuration. `min_available_gib` reserves free system memory. Lowering a reserve does not create memory. Process guards are cooperative sampling and cannot guarantee prevention of an OS out-of-memory event.

**GPU memory is `null`.** NVIDIA process reporting may be unavailable due to device/driver mode, permissions or lack of support. Apple unified memory has no fabricated dedicated-VRAM measurement. External endpoints have no managed process telemetry. Unknown is not zero; remove a GPU gate only if that decision is appropriate to your evaluation and retain the limitation in the result.

**Context or output limit.** Candidate validation only requires context larger than the output limit. Your prompt and chat template also need room. Reduce context input, increase the supported context, or narrow the task. Inspect `truncated_requests` and raw responses. An output budget that cuts off valid structured responses is not a fair quality comparison.

**Structured answers score zero.** JSON scoring expects the whole stripped answer to parse, with no prose or Markdown fence. Check extra keys, value types and requested fields. Where supported, a supplied `json_schema` and candidate `constrain_json` can enforce output structure; retain the same task set for constrained/unconstrained comparisons. Grammar does not establish semantic correctness.

**Throughput missing.** The backend did not supply all required token counts or timings. External streaming services may provide usage but no decode timing. Keep elapsed latency as measured and report unavailable throughput honestly.

**Resume rejected.** Inputs, configuration or runtime changed, or the experiment uses an external endpoint. Preserve the old record and use a new output directory. Managed resume retries incomplete trials; it does not continue from a partial trial's last task.

## Routing and provider requests

**No feasible model.** Read per-model rejection reasons. Automatic cost needs both prices; performance needs latency; balanced needs cost, latency and quality. Unknown is not free or compliant. Pin a suitable model for an initial check, provided it still passes hard constraints, or enter calibrated metrics.

**Pinned model rejected.** Pinning does not override placement, capability, context, output or budget constraints. Check that `--model` is the registry ID, not an unrelated provider model name.

**Key absent.** Set the environment variable whose name appears in `api_key_env` before launching the CLI/UI. A key set in another terminal or after startup is not inherited. Cloud models require a populated key. Never paste the key value into registry JSON to work around this.

**HTTP 400/401/404 or malformed provider response.** Verify the exact model ID, credentials, API base path and supported request fields. The app appends `/chat/completions` or `/messages`; do not include those suffixes twice. OpenAI compatibility varies, particularly around `max_completion_tokens`, streaming usage and finish markers. Check server logs locally without publishing private prompt/key contents.

**Local placement rejects a destination.** Its DNS addresses must be loopback/private and not unspecified/multicast. A public inference service belongs under cloud placement and HTTPS. Redirects are deliberately not followed.

**Container cannot reach the host model.** Container loopback is not host loopback. Use a reachable private gateway address, such as the configured `host.docker.internal`, and confirm the host inference server's bind address. See [Docker networking](docker.md).

**Cancelled external request still completes.** Cancellation is cooperative. An already-sent provider call can finish before cancellation is observed, and its provider may still process or charge for it. Do not infer that cancelling the GUI cancelled billing, and do not assume an uncertain result should be automatically repeated. The app has no hidden retry/fallback.

## Code proposals and containers

**Malformed code JSON.** The model must return the `summary`/`files` contract with complete file text. Ask for a smaller change or choose an endpoint/model that follows structured instructions reliably. Prose, arbitrary patches, extra keys or truncated responses can be rejected before apply.

**File changed since preview.** Refresh context and generate a fresh proposal. Do not manually remove the original-file hash check to apply a stale diff.

**Path rejected.** Use ordinary portable project-relative paths. Hidden paths, symlinks, traversal, absolute paths, case-variant duplicates and Windows reserved filenames are outside this development workflow.

**Docker CLI unavailable or engine stopped.** Start Docker and run the feature through the host lab/CLI. The small application image does not include Docker access.

**Image unavailable.** Choose **Allow image pull** or `--pull` for an intentional first download, or pre-load the image. Test-container network permission and image-pull permission are separate.

**Dependencies missing.** Default Python/Node images do not install project packages. Use a prepared compatible image with CLI `--image`. Hidden files and `node_modules` are excluded from project copying; inspect the copied workspace.

**Permission failure.** Execution uses a non-root identity. Check that the copied workspace and prepared image support it; on Docker Desktop confirm host file sharing is available. Do not solve it by giving generated code a Docker socket or privileged host mount.

**Container passes but did nothing useful.** Read the command, stdout and test count. Python build is compilation only; a runner discovering zero tests is not functional coverage. Add meaningful tests for the behaviour you care about.

For an issue report, include the application/Python/runtime versions, OS/architecture, a small sanitised configuration, exact command, relevant error and logs. State whether evidence comes from mocks, local hardware, an external provider or Docker. Exclude credentials and private task data.
