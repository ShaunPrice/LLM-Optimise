# User guide

Optimisation starts with a task that matters. The lab measures whether a configuration meets your task's quality requirement, then compares the eligible configurations on latency and memory. It does not assume that the smallest model, highest tokens per second, or lowest weight precision is the best result.

## Design an experiment you can trust

### Define the specialised task and acceptance rule

Write down the intended inputs, required outputs, errors that matter and an acceptable quality threshold before running a sweep. For example, a sensor-event classifier might require a fixed label, while a document extractor might require an exact JSON schema and correct values. Include difficult, ambiguous and negative cases from the intended workload.

Keep training data, development evaluation data and a final held-out test split separate. Use the development split to choose quantisation, context and runtime settings. Evaluate the selected configuration once on the held-out split before making a deployment decision. Repeatedly selecting settings against the final test set weakens its independence.

Use the same dataset, prompt template, output limit, repeats and scoring rule for configurations you compare. If you change the dataset or task definition, create a new experiment and treat the result as a new comparison.

### Prepare the evaluation JSONL

Each nonblank line is a JSON object. IDs must be unique. `id`, `prompt` and `expected` are required; `evaluator`, `system`, `tolerance` and `json_schema` are optional.

```jsonl
{"id":"alarm-label","prompt":"Classify: motor temperature exceeds the configured safety threshold. Return alarm or normal.","expected":"alarm","evaluator":"exact"}
{"id":"sensor-count","prompt":"Return only the number of sensors in: three inlet sensors and two outlet sensors.","expected":5,"evaluator":"numeric","tolerance":0}
{"id":"extract","prompt":"Return JSON with sensor and state: inlet-2 reports high temperature.","expected":{"sensor":"inlet-2","state":"high_temperature"},"evaluator":"json"}
```

These rows illustrate the format. They are not a sufficient validation set for an operational system.

| Evaluator | Acceptance rule |
| --- | --- |
| `exact` | Trim leading/trailing whitespace and compare case-insensitively. Extra explanation fails. |
| `numeric` | Parse the entire stripped answer as a finite number and compare using an absolute `tolerance`, default `0`. |
| `json` | Parse the entire answer as JSON and compare values and structure. Additional object keys fail; object key order does not matter. |
| `json_subset` | Require the expected nonempty object and its nested values; additional object fields are allowed. |

Boolean JSON values are distinguished from numeric `0` and `1`. A JSON answer wrapped in prose or Markdown does not pass JSON scoring. Each request receives `0` or `1`; aggregate quality is their arithmetic mean. The evaluator never executes generated code. Use separate meaningful code tests for programming tasks.

For structured-output experiments, add a `json_schema` describing the required output shape, then compare candidates with `constrain_json: false` and `constrain_json: true`. The managed adapter passes that schema to llama.cpp; the external adapter uses the compatible structured-output request format when supported. A schema should describe field names and types, not encode each task's expected answer. Keep the same dataset and scoring rule for both candidates so a formatting improvement is measured fairly. A valid JSON object can still contain an incorrect answer.

### Start with one candidate

Save the following as `examples/domain-baseline.json`, provide `examples/domain-eval.jsonl`, and replace the model path with your actual model:

```json
{
  "name": "domain-baseline",
  "dataset": "domain-eval.jsonl",
  "repeats": 3,
  "warmup": 1,
  "max_tokens": 64,
  "timeout_s": 120,
  "startup_timeout_s": 120,
  "max_trials": 8,
  "seed": 42,
  "limits": {
    "min_quality": 0.9,
    "max_rss_gib": 4,
    "min_available_gib": 1
  },
  "candidates": [{
    "name": "cpu-baseline",
    "model": "../models/my-model.gguf",
    "executable": "llama-server",
    "threads": 4,
    "context": 2048,
    "gpu_layers": 0,
    "batch_size": 256,
    "ubatch_size": 64,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "flash_attention": "off"
  }]
}
```

The memory values are example budgets, not a claim that your model fits. Set them from your available hardware and reserve memory for the operating system and other applications.

```bash
llm-optimise validate examples/domain-baseline.json
llm-optimise run examples/domain-baseline.json --output runs/domain-baseline
```

Candidates start sequentially in a seeded shuffled order. Tasks are also shuffled deterministically for each repeat. Repeats sample execution variability; they do not create new independent task examples. Keep other heavy workloads idle and record relevant power, cooling, runtime and driver settings.

### Expand a controlled sweep

After a successful baseline, add a `sweep` to the candidate in a new configuration:

```json
"sweep": {
  "threads": [2, 4],
  "context": [1024, 2048],
  "gpu_layers": [0, -1]
}
```

This produces eight candidates. With six tasks and three repeats, it makes 144 measured requests, plus warmups. `max_trials` bounds the expanded Cartesian product before model execution.

Use `gpu_layers: 0` for explicitly forced CPU execution, `-1` for all supported layers, or a positive count for partial offload. Accelerator support is determined by the runtime. A GPU request does not establish that GPU execution succeeded; inspect runtime logs and measurements.

Useful follow-up comparisons include weight files at different quantisation levels, context size, thread count, batch and microbatch sizes, KV cache precision and offload. Compare only settings supported by the selected runtime. Quantised V cache requires `flash_attention: "on"`. Keep `ubatch_size` no larger than `batch_size`.

Changing several dimensions at once finds candidates quickly but makes causal explanation harder. Follow the broad sweep with a small comparison that changes one setting at a time around the best feasible candidates.

## Read the measurements

| Field | Meaning and practical limit |
| --- | --- |
| `load_s` | Time to start the managed server and reach readiness, measured separately from requests. OS file cache can make later loads faster; this is not guaranteed cold-disk loading. |
| `quality` | Mean deterministic request score across tasks and repeats. Read `quality_by_task` to find concentrated failures. |
| `latency_p50_s`, `latency_p95_s` | Median and interpolated 95th percentile of measured request latency. Small samples make the tail estimate weak. |
| `ttft_p50_s`, `ttft_p95_s` | Time until first nonempty output content, including prompt preparation. Missing first content yields missing TTFT. |
| `decode_tokens_s` | Aggregated backend-reported decode throughput when all required token counts and timings are present. |
| `end_to_end_tokens_s` | Total reported output tokens divided by total request latency. Includes more than decode time. |
| `rss_peak_gib` | Sampled peak RSS of the managed server process tree. It is not the whole machine's memory use. |
| `gpu_peak_gib` | Supported NVIDIA process memory samples, explicit zero for forced managed CPU execution, otherwise potentially unavailable. |
| `system_available_min_gib` | Lowest sampled available system RAM during the trial. |
| `truncated_requests` | Outputs reported as truncated or stopped at their token limit; inspect them before accepting the configuration. |

Warmup uses the first dataset task and is excluded from scored request samples. Memory monitoring spans server startup, warmup and measurement. Model loading and request latency answer different questions; do not combine them without stating the workload assumption.

Missing values remain `null` in JSON and appear as unavailable in the interface. An external endpoint has no managed process PID, so its server RSS and GPU memory are unavailable. A response without usage cannot support token-based throughput accounting. Streaming OpenAI-compatible endpoints normally do not provide llama.cpp's native decode timing.

On Apple Silicon, CPU and GPU share physical memory. Process RSS is not a measurement of peak Metal allocations, and the lab does not fabricate dedicated Apple VRAM. Do not add a supposed Apple GPU allocation to system RAM as though they were independent pools. A strict GPU-memory gate cannot pass if its required measurement is unavailable.

## Understand eligibility and the frontier

A trial must finish and satisfy every configured quality, memory and latency condition to be eligible. An unavailable observation fails an explicitly requested bound. The report explains each rejection; a completed experiment can correctly have no frontier.

The Pareto frontier contains eligible candidates that no other comparable candidate dominates across quality, median latency, peak process RAM and peak GPU memory. A candidate dominates another when it is at least as good in every observed dimension and strictly better in one. Different missing-telemetry patterns are not treated as proof of superiority.

Several frontier candidates may be useful. Choose among them using your real deployment priority, such as latency for interactive work or RAM reserve on a shared machine. A frontier is relative to the candidates and task data tested; it is not evidence of a global optimum.

Inference memory guards sample and cooperatively stop execution after a violation. They are not operating-system hard caps and can miss short peaks or stop after memory has already been allocated. Docker project execution provides a separate mechanism with engine-enforced resource limits; see [Docker](docker.md).

## Use the graphical views

**Experiment lab** supports a baseline, small sweep, custom fields and advanced JSON. Enabling advanced JSON replaces the form configuration. GUI relative paths resolve from the workspace. Select a saved result, change the chart dimension, filter eligible trials, inspect trial details and export result JSON. Standalone HTML and CSV reports are also written to the run directory.

**Model router** registers providers and previews a selection without inference. Its placement, objective and hard constraints also govern Chat and Develop. Set the desired budgets in the routing controls and inspect rejection reasons before sending a request. See [routing](agent-routing.md) for metric requirements and defaults.

**Chat** receives detected hardware, available task/model paths and compact summaries of up to two recent result sets, including at most twelve trials per result. Ask it to explain a tool, interpret a quality failure, or propose the next experiment. The configured provider receives this context. A model's explanation is still generated advice and should be checked against artifacts.

Valid chat suggestions use known local model, runtime and dataset paths, at most eight trials, three repeats and 128 output tokens. The assistant cannot execute a shell command. Load a suggestion into the lab to inspect it, then choose whether to run it. Stop a managed chat model before benchmarking.

**Develop** generates and applies reviewed file proposals. **Training** prepares Soup configuration and a command without starting training. See [development](development.md) and [training research](training-research.md).

## Save, reproduce and resume

| Artifact | Purpose |
| --- | --- |
| `runs/<run>/results.json` | Experiment settings, hardware snapshot, file hashes, runtime details, trial status, metrics and frontier |
| `runs/<run>/trial-*.jsonl` | Measured request text, timing, task ID, repeat and score |
| `runs/<run>/trial-*.log` | Managed runtime output for load and accelerator diagnosis |
| `runs/<run>/report.html` and `results.csv` | Portable presentation and tabular export |
| `.llm-optimise/experiments/` | GUI-generated experiment configurations |
| `.llm-optimise/models.json` | User model catalogue; credentials are referenced by environment variable name |
| `runs/jobs/`, `runs/recipes/`, `runs/containers/` | GUI operation records, prepared training recipes and container artifacts |
| `projects/<name>/` | GUI development projects |

Request artifacts may contain your evaluation prompts, expected task context and model responses. Review artifacts before publishing or sharing them.

Managed runs fingerprint the expanded experiment plus dataset, model, optional draft model and discoverable runtime executable hashes. Resume skips completed trials and retries incomplete ones, not individual requests within a partial trial:

```bash
llm-optimise run examples/domain-baseline.json --output runs/domain-baseline --resume
llm-optimise report runs/domain-baseline/results.json --output runs/domain-baseline
```

Changed inputs or runtime reject resume. External endpoint runs cannot resume because a remote server's model identity may change. Preserve the original configuration and artifacts, use a fresh output directory for changed experiments, and record external model/runtime identity independently.

The CLI returns `2` when a run is incomplete or has no eligible frontier. This is useful in automation and is not interchangeable with an application configuration error. See [CLI reference](cli-reference.md).


## Experimental workbench additions

The [Workbench guide](workbench.md) documents executable dataset, adapter, search, calibration, caching, context, distillation, component, repair and model-lifecycle workflows shared by the GUI and CLI. Native process supervision and desktop packaging are documented under [native](../native/README.md) and [desktop](../desktop/README.md).
