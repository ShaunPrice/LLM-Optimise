# CLI reference

Run `llm-optimise --help` or append `--help` to any subcommand for the installed version's exact options. All examples assume the application is installed in your active virtual environment. On Windows without activation, invoke `.\.venv\Scripts\llm-optimise.exe`.

Most commands emit JSON on stdout. Experiment progress is written separately to stderr. Paths containing spaces should be quoted. JSON uses finite numbers and `null` for unavailable values.

## Commands at a glance

| Command | Purpose | Loads or calls a model? |
| --- | --- | --- |
| `doctor` | Detect hardware and runtime executables | No |
| `ui` | Serve the local browser application | Only after an explicit GUI operation |
| `validate` | Check/expand experiment configuration and load task definitions | No |
| `estimate` | Calculate approximate weight and KV payload memory | No |
| `run` | Execute a bounded experiment | Yes |
| `report` | Export an existing result to HTML and CSV | No |
| `recipe` | Prepare a Soup training configuration | No |
| `route` | Explain a policy-based model choice | No |
| `agent` | Make one routed text request | Yes |
| `code` | Make one request and prepare a code proposal | Yes |
| `apply` | Apply a reviewed proposal | No |
| `container` | Execute a copied project in Docker | No inference; explicitly runs project code |

## `doctor` and `ui`

```bash
llm-optimise --version
llm-optimise doctor
llm-optimise ui --workspace . --port 8765 --open
```

`doctor` reports OS, architecture, CPU counts, available/total RAM, detected NVIDIA devices, unified-memory status and paths for `llama-server`, `soup` and `mlx_lm.server` when found. Runtime detection is not a live model test.

`ui` defaults to workspace `.`, port `8765`, and host `127.0.0.1`. `--open` asks the default browser to open the lab. `--host 0.0.0.0` supports container ingress; the supplied Compose configuration publishes only to host loopback. The app is designed for local use and validates browser Host/Origin.

## `validate`, `run` and `report`

```bash
llm-optimise validate examples/my-experiment.json
llm-optimise run examples/my-experiment.json --output runs/my-run
llm-optimise run examples/my-experiment.json --output runs/my-run --resume
llm-optimise report runs/my-run/results.json --output runs/my-run
```

`validate CONFIG` reports candidate count, task count and measured request count. It does not load the runtime/model. `run CONFIG --output DIRECTORY` requires an empty/new directory unless `--resume` is supplied. Managed resume requires unchanged configuration and input/runtime hashes. External endpoint runs do not support resume.

`report RESULTS --output DIRECTORY` reads an existing `results.json` and writes `report.html` and `results.csv`. It does not repeat inference.

### Experiment JSON

Unknown configuration keys are rejected. See [the tutorial](user-guide.md) for a complete file.

| Experiment key | Default | Meaning |
| --- | --- | --- |
| `name`, `dataset`, `candidates` | Required | Label, task JSONL path, candidate list |
| `repeats` | `3` | Measured passes through every task |
| `warmup` | `1` | Unscored warmup requests using the first task |
| `max_tokens` | `64` | Per-request output limit |
| `timeout_s` | `120` | Request timeout |
| `startup_timeout_s` | `120` | Managed server readiness timeout |
| `max_trials` | `32` | Expanded candidate bound |
| `seed` | `42` | Request seed and deterministic ordering |

| Candidate key | Default | Meaning |
| --- | --- | --- |
| `name`, `model` | Required | Unique name; local GGUF path or served model ID |
| `backend` | `llama.cpp` | `llama.cpp` or `openai` |
| `executable` | `llama-server` | Managed runtime executable |
| `endpoint` | `null` | Required API base URL for external `openai` backend |
| `threads`, `context` | `4`, `2048` | CPU threads and context tokens |
| `gpu_layers` | `0` | `0` CPU, `-1` all supported, positive count for partial offload |
| `batch_size`, `ubatch_size` | `256`, `64` | Logical and physical batch bounds |
| `cache_type_k`, `cache_type_v` | `f16`, `f16` | `f16`, `q8_0` or `q4_0` |
| `flash_attention` | `off` | `off`, `on` or `auto`; quantised V requires `on` |
| `mmap`, `cache_prompt` | `true`, `false` | Managed model mapping and prompt-cache request setting |
| `constrain_json` | `false` | Send a task's supplied JSON schema through the selected backend's structured-output interface; server support is required |
| `draft_model` | `null` | Optional compatible managed draft GGUF path |
| `api_key_env` | `null` | Name of an environment variable for an external endpoint key |
| `sweep` | Absent | Map model/inference settings to nonempty lists; expands as a Cartesian product |

Candidate context must exceed `max_tokens`. That basic validation does not prove every prompt plus output fits. Relative CLI dataset, model and executable paths resolve from the config file directory; a bare executable name is looked up on `PATH`.

`limits` defaults to `min_quality: 0.9` and `min_available_gib: 1`. Optional bounds are `max_rss_gib`, `max_gpu_gib`, `max_latency_s` and `max_ttft_s`. Latency gates use measured p95. An unavailable required measurement rejects eligibility.

### External endpoint experiment

An external benchmark candidate has this shape; replace the model and endpoint with your actual service:

```json
{
  "name": "existing-server",
  "model": "model-served-by-your-runtime",
  "backend": "openai",
  "endpoint": "http://127.0.0.1:8080/v1",
  "context": 4096
}
```

The server must already be running and support the adapter's streaming completion request. Runtime settings such as `threads` and `gpu_layers` do not reconfigure an external server. Process memory is unavailable; record its runtime, model version and deployment settings separately. Explicit external experiments can contact the endpoint you configure; chat-generated experiment suggestions are limited to known local managed models.

## `estimate`

```bash
llm-optimise estimate --parameters-b 1.5 --weight-bits 4.5 --layers 28 --kv-heads 2 --head-dim 128 --context 2048 --kv-bits 16 --concurrency 1 --overhead-gib 0.5 --unified
```

Options/defaults: `--parameters-b 1.5`, `--weight-bits 4.5`, `--layers 28`, `--kv-heads 2`, `--head-dim 128`, `--context 2048`, `--kv-bits 16`, `--concurrency 1`, `--overhead-gib 0.5`, `--gpu-fraction 1`; `--unified` is off by default.

These numbers illustrate the calculator, not a recommended model. Use actual architecture metadata. The estimate covers dense full-attention transformer weight/KV payloads and a chosen overhead allowance. It does not model training, MoE, hybrid/SSM, MLA, sliding-window attention or all runtime allocations, and is not a fit guarantee.

## `recipe`

```bash
llm-optimise recipe --engine soup-stream --model models/my-training-model --data data/train.jsonl --model-output adapters/domain --max-length 512 --rank 8 --output runs/recipes/domain-stream.yaml
```

Required: `--engine` (`soup-stream`, `soup-qlora`, `soup-mlx`), `--model`, `--data`, `--output`. Defaults: `--model-output ./adapters`, `--max-length 512`, `--rank 8`.

The output is JSON-valid YAML plus printed notes and the command `soup train --config ...`. It prepares one-epoch SFT configuration with Alpaca-format training data, batch size one, LoRA, four-bit quantisation and gradient checkpointing. Stream and MLX options differ; read the generated notes and [training research](training-research.md).

Recipe creation does not validate the training dataset/model, install Soup, download weights, or run training. Use a separate Python 3.10–3.12 Soup environment and an independent held-out evaluation split. `soup` extras in this project's packaging are not an installation of the full training stack.

## `route`, `agent` and `code`

These commands share the following options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--models FILE` | Required | JSON catalogue |
| `--placement` | `local` | `local`, `cloud`, `mixed` |
| `--objective` | `cost` | `cost`, `performance`, `balanced` |
| `--model ID` | Automatic | Pin a catalogue ID while retaining hard constraints |
| `--max-cost-usd` | Unset | Maximum estimated request price |
| `--max-latency-ms` | Unset | Maximum catalogue latency estimate |
| `--min-quality` | Unset | Minimum catalogue quality |
| `--max-ram-gib`, `--max-gpu-gib` | Unset | Maximum local model memory estimates |
| `--max-tokens` | `1024` | Requested maximum output |

`route` adds `--input-tokens` (default `1000`) and `--capability` (default `chat`). Input/output token counts must be positive.

```bash
llm-optimise route --models .llm-optimise/models.json --placement local --objective cost --input-tokens 1000 --max-tokens 256 --capability code --max-ram-gib 4
```

`agent` requires `--message`; `--output FILE` optionally saves the result. This is a single routed text request, not the GUI's context-enriched lab conversation.

```bash
llm-optimise agent --models .llm-optimise/models.json --model local-demo --message "Explain the difference between load time and request latency." --max-tokens 256 --output runs/agent-answer.json
```

`code` also requires `--project` and optionally accepts `--context` followed by project-relative filenames. It uses the `code` capability and adds a validated proposal to the result. See [development](development.md).

## `apply` and `container`

```bash
llm-optimise apply runs/proposal.json --project projects/my-project
llm-optimise container --project projects/my-project --output runs/my-project-tests --runtime python --action test
```

`apply PROPOSAL --project DIRECTORY` accepts the whole saved code result or its proposal object. It checks original file hashes and refuses already-applied or stale proposals.

`container` requires `--project` and `--output`. Options/defaults: `--runtime python` (`python`/`node`), `--action test` (`test`/`build`), `--memory-mib 512`, `--cpus 1`, `--timeout-s 60`; `--network` and `--pull` are opt-in, and `--image` overrides the runtime's default image. See [Docker](docker.md) for exact commands and dependency requirements.

## Exit statuses

| Code | Meaning |
| --- | --- |
| `0` | Command success; for experiments, complete with at least one eligible frontier candidate |
| `1` | Handled application/configuration/provider error |
| `2` | Incomplete/no-frontier experiment, failed/timed-out container action, or argument-parser usage error |
| `130` | Keyboard interrupt handled by the CLI |

An interrupt handled inside the experiment runner can instead produce an incomplete result and exit `2`. Preserve stdout/stderr and the result record when diagnosing an automated run.
