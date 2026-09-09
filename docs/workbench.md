# Workbench: build, measure and specialise

Open **Workbench** in the application. Choose an area, select a tool, fill in its guided form, and inspect the request preview. Long operations appear in the shared job list with progress and cancellation. Expand the result to inspect the recorded evidence and artifact directory. Advanced JSON is available for reproducible requests.

The same operation names work through the CLI:

```bash
llm-optimise workbench status --workspace .
llm-optimise workbench dataset-prepare --request examples/workbench/dataset.json --workspace . --output runs/domain-v1
llm-optimise workbench dataset-evaluate --request examples/workbench/evaluate.json --workspace .
llm-optimise explore examples/workbench/capacity.json --plan --workspace .
```

Paths in workbench requests are resolved against `--workspace`. Use a fresh output directory for runs. Generated projects live under the workspace's `projects/` directory; the workbench `project` field is that project's name. Cloud credentials remain environment variables on the service host. Artifacts, model registries, caches and datasets are local and ignored by Git.

## Data and quality

**Prepare dataset** accepts JSON/JSONL rows containing `id`, `prompt`, `expected` and an evaluator: `exact`, `numeric`, `json` or `json_subset`. Optional fields include `system`, `tolerance`, `json_schema`, `task_class`, `group`, `error_cost` and `schema_error_cost`.

Preparation deduplicates identical prompt/system pairs, rejects conflicting labels, and keeps groups together across train, tuning and held-out partitions. Supply a `group` for documents, conversations or source families that must not leak across splits. The manifest records hashes, split counts and warnings; tiny datasets can produce empty partitions. Strict task files feed the experiment runner; metadata row files preserve grouping and error costs for evaluation and distillation; supervised files feed training.

**Evaluate predictions** joins predictions to task IDs, reports missing/wrong/schema-invalid answers, and retains per-task results and weighted error costs. **Compare regression** checks the candidate against the baseline using explicit gates. Comparisons require compatible dataset identities. A schema is checked using the documented supported subset: types, object properties/required/additionalProperties, array items, enum/const and numeric/string/array bounds. Unsupported schema keywords are rejected.

Use the tuning split to choose configurations. Reserve held-out data for a final check. Six bundled smoke prompts only establish that the plumbing works.

## Training and adapters

**Probe environment** checks the explicitly selected Python executable. **Run training** launches real Soup SFT in that separate environment, using a downloaded local model and dataset. Select MLX on Apple Silicon or Transformers for supported CPU/CUDA environments. The core application does not install an ML framework or download a model on launch.

Training accepts bounded steps, process RSS, optional supported GPU telemetry, free-memory reserve and wall time. Output includes runtime versions, Soup source identity, model/data hashes, configuration, logs, resource observations and saved adapter files. Hugging Face offline flags are enabled by default; they are not an OS network sandbox. Use reviewed environments and data.

Successful training registers the adapter. **Register adapter** imports an existing compatible adapter with base-model provenance. **Reload adapter** checks that its files and base model still match, starts a fresh worker and performs a smoke generation. **Evaluate adapter** runs task rows, optionally using `baseline: true` to evaluate the base model. **Compare adapters** applies the regression gates to recorded evaluations. Model/backend/adapter compatibility is checked; a file existing on disk is insufficient evidence that its weights were applied.

Transformers adapter evaluation currently loads the base in its default dtype; it does not reconstruct a training-time BitsAndBytes configuration. Budget RAM accordingly. MLX retains the downloaded model quantisation.

A minimal training request has this shape; change paths and runtime settings for your environment:

```json
{
  "python": "/absolute/path/to/soup-environment/bin/python",
  "max_steps": 20,
  "timeout_s": 600,
  "max_rss_gib": 4,
  "min_available_gib": 1,
  "config": {
    "base": "models/downloaded-huggingface-model",
    "task": "sft",
    "backend": "mlx",
    "data": {"train": "data/train.jsonl", "format": "chatml", "max_length": 128, "val_split": 0},
    "training": {"epochs": 1, "batch_size": 1, "lr": 0.0002, "lora": {"r": 4, "alpha": 8}, "quantization": "4bit"}
  }
}
```

Soup streaming requires a compatible Transformers/CUDA setup and layer-source settings; it is a comparison option, not a promised improvement. See [training research](training-research.md) and [validation](validation.md).

## Hardware exploration

All five explorer modes use the same local experiment specification, task scoring and sampled memory limits. First plan the run; unavailable flags/devices are excluded with reasons. No accelerator is assumed from its name alone.

| Mode | Behaviour |
|---|---|
| `capacity` | Increase configured context/batch/thread dimensions in bounded steps; retain the first failing boundary, stop higher steps, wait for headroom and optionally rerun the last passing control. |
| `kv` | Compare context lengths and f16/q8_0/q4_0 KV settings; quantised KV requests Flash Attention. |
| `accelerators` | Compare the same model/task contract through explicit CPU, Metal, CUDA, Vulkan or ROCm runtime/device choices that the installed executable reports. |
| `speculative` | Compare ordinary decoding with local draft models and bounded draft lengths; retain draft-token/acceptance metrics when the runtime exposes them. Missing counters remain null. |
| `halving` | Run common nested development-task subsets, promote eligible candidates to larger task budgets, and evaluate the selected candidate once on a disjoint optional held-out dataset. |

Use `max_duration_s`, `max_trials`, request/startup timeouts and a positive `min_available_gib` reserve. Sampled process RSS cannot guarantee detection of brief allocation spikes. Apple unified memory uses system headroom and RSS; unobservable Metal allocation is null. CPU zero GPU use and unknown GPU use are distinct. Installed backends, supported model formats and draft compatibility remain hardware/runtime requirements.

The results contain failures, exclusions, selection reasons, a feasible frontier and uncertainty notes. A capacity result is the limit reached by that bounded configuration sweep under its recorded workload; it is not a universal maximum for the device.

## Routing, caching and context

**Calibration** makes explicitly budgeted calls over scored task data. Observations are scoped by task class, model and adapter identity, input-size bucket and hardware identity. With empirical routing enabled, quality uses the 95% Wilson lower bound for binary task success; unscored chats do not establish quality. Fewer than three relevant samples leave a metric unknown. Calibration retains partial work on failures and stops further calls when its remaining estimate is exhausted. Actual provider charges may exceed estimates.

The bounded observation store evicts older records from the most populated scope first. Evidence expires from routing after 30 days. Manual registry estimates remain available when calibration is disabled.

**Exact result caching** is opt-in and requires an explicit model revision. The key covers endpoint/model/adapter identity, full messages, output limit, schema, decoding policy and prefix-cache option. Cached calls use temperature zero, expire after one day by default, and occupy at most 32 MiB of payload storage. Hits do not create provider charges or new latency/quality observations. Placement and other routing constraints still apply before a hit is served. Cache status and clearing are available in the interface.

**Prefix caching** requests compatible local OpenAI-style servers to reuse their prompt prefix. Actual reuse is controlled by the runtime. It does not imply arbitrary cloud provider cache support. Provider-reported cached input tokens are recorded when present.

**Context selection** ranks relevant source chunks within a character budget, reports line spans/source hashes and checks explicit required facts. Irrelevant chunks are omitted; missing required facts block generation when selection is enabled in Develop. Full file snapshots remain the basis for stale-file protection. This is lexical retrieval, not a guarantee of semantic completeness. CLI code generation supports `--context-max-chars` and repeatable `--required-fact`.

## Specialists and distillation

Register `task_classes` on specialised models to restrict them to their intended tasks. **Specialist response** requires an exact allowed-output list or a supported JSON schema. It returns an accepted result or abstention. A second model is used only when explicitly selected as the escalation model and covered by a total cost budget; placement and task constraints still apply.

**Distillation** sends authorised training-split prompts to the selected teacher, with request/token/time/cost limits. Reference answers and schema gates curate labels; rejected outputs remain in the audit. Ungated labels require explicit selection and do not establish correctness. Held-out/tuning rows are rejected. Unknown cost stops subsequent calls. A zero-dollar budget admits only configured models with known zero prices. Review the resulting labels before training and evaluate the student on unseen data.

## Development and deterministic components

**Repair loop** works on a disposable project copy and uses the selected model to propose changes. It runs bounded Docker checks, returns failures to the model within iteration/time/cost limits, and produces a final reviewable proposal. The original project changes only when that proposal is applied. Tests, harnesses, build configuration and their byte hashes are protected from generated edits.

Python acceptance uses a fixed isolated unittest runner and a structured report; Node uses its built-in JUnit reporter with explicit test files. A printed “tests passed” message cannot establish success. Existing projects that depend on pytest-specific plugins or custom test frameworks should use the normal container workflow or adapt a reviewed acceptance harness. Rust repair uses explicit host-scored component cases. These controls catch accidental test weakening; they are not formal proof against adversarial code executing in the same container.

**Deterministic component** builds/runs a Python or Rust JSON-in/JSON-out program against explicit cases. Expected outputs are scored on the host and are not supplied to the program. Each case uses a fresh constrained container. Rust is compiled separately; execution reports include cold process/container overhead and labelled resource scope. The component can replace LLM calls for precisely specified transformations only after the acceptance cases are representative.

Both operations require a reviewed execution action. Network is disabled during acceptance runs; images must be installed already or explicitly pulled. Use CPU, memory, PID and time limits. Docker remains a separate prerequisite. See [Docker](docker.md).

## Model residency and native components

Managed models use memory admission, per-model/global request limits, deduplicated loading and idle unloading. A request acquires a residency lease and reloads an unloaded registered model when needed. Changing a model file requires registering its new identity. Admission estimates and observed process caps are separate; unknown explicit GPU caps fail closed.

Choose an optional Rust supervisor executable in the managed-model or explorer settings. Build it with `cargo build --release --manifest-path native/Cargo.toml`. Python delegates process ownership, sampled RSS and cancellation through a bounded JSONL protocol. An uncertain native launch does not silently start a second Python worker. [Native implementation and measured profile](../native/README.md).

The [Tauri desktop shell](../desktop/README.md) provides a native launcher, window, icon and tray. It uses the same GUI/service and your selected Python environment. Closing the lab hides it; tray Quit stops the owned service/models. Development bundles require platform-specific build tools; release signing/notarisation and installer acceptance are separate from compilation.

## API and artifacts

`POST /api/workbench/<operation>` accepts the same request object as the CLI. Long operations return a job ID; poll the shared state/jobs endpoint and use the normal cancellation control. Mutating API calls require the application's same-origin session token. Lifecycle settings and unloading use `/api/lifecycle/settings` and `/api/lifecycle/unload`.

Operation names: `explore-plan`, `explore-run`, `dataset-prepare`, `dataset-evaluate`, `dataset-compare`, `training-probe`, `training-run`, `adapter-register`, `adapter-reload`, `adapter-evaluate`, `adapter-compare`, `distill`, `calibrate`, `context`, `feedback`, `component`, `specialist`, `cache-clear`, `status`.

Outputs normally live under `runs/workbench/<job-id>/`. Cache/calibration data and adapter registration live under `.llm-optimise/`. Reports can contain prompts, source excerpts, model outputs and local paths; review artifacts before sharing them. API keys are referenced by environment-variable name and are not stored in these records.
