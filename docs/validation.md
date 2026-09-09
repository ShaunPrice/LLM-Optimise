# Validation evidence

Recorded 9 September 2026. Software checks, live model execution and untested capabilities are distinguished below.

## Live inference: Apple M4

Host: macOS, Apple M4, 24 GiB unified memory. Runtime: official llama.cpp macOS ARM64 build 10873, commit `6de9cdb26`, reporting `0.4.0-dev`. Model: `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf`, obtained from [bartowski's repository](https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF/tree/1af47f78b1f9b0c242fabe43f7a365d5a67f3207). Weights are not included here.

The configuration compares CPU-only (`gpu_layers: 0`) and Metal offload (`-1`) with JSON schema constraints off/on. Each candidate runs in a new managed process, with one warmup and three measured repetitions of each of six smoke tasks. Context is 2048, threads 4, output limit 48 tokens. The quality gate is 0.9. Candidate execution order is seeded and recorded. See [configuration](../examples/coder-constrained.json), [dataset](../examples/tasks.jsonl), [results JSON](../validation/m4-constrained/results.json), [standalone HTML](../validation/m4-constrained/report.html) and [CSV](../validation/m4-constrained/results.csv).

| Execution | JSON constraint | Quality | Median latency | p95 latency | Decode tokens/s | Peak sampled RSS |
|---|---:|---:|---:|---:|---:|---:|
| Metal | Off | 83.3% | 117 ms | 434 ms | 59.3 | 1.11 GiB |
| CPU | On | 100% | 241 ms | 451 ms | 60.2 | 2.00 GiB |
| Metal | On | 100% | 98 ms | 283 ms | 64.2 | 1.12 GiB |
| CPU | Off | 83.3% | 247 ms | 576 ms | 58.7 | 2.02 GiB |

The unconstrained model wrapped JSON in Markdown, which failed the strict extraction evaluator. The schema specifies structure without supplying the answer. Constraints repaired this failure in this dataset. This is a small workflow validation; there is no significance claim and no assertion that it generalises to other tasks.

Decode throughput is taken from llama.cpp generation timing and weighted by decoded token count. It excludes model loading and prompt evaluation. The qualifying Metal configuration's end-to-end throughput was 29.9 tokens/s; median first-token latency was 81.6 ms. Request latency is measured by the client. These metrics answer different questions.

RSS is sampled every 50 ms for the managed process tree and can miss short peaks. It includes neither all system allocations nor a complete attribution of Metal allocations. Metal GPU memory is unknown, not zero. CPU-only explicitly records zero accelerator use. The Pareto calculation does not dominate a candidate across incompatible missing-metric dimensions, so both qualifying CPU and Metal candidates are retained. Shared Apple memory is not added to RAM as though it were a separate device pool.

Paths in the published JSON are replaced with `<workspace>`; hashes, numerical observations and trial order are retained. The fingerprint refers to the original run. To reproduce, download the licensed model, install a compatible runtime, update the executable path, and use a new output directory. Hardware, thermal state, background load and runtime versions affect results.

## Software and integration checks

- 148 local automated tests pass, including routing constraints, unknown metrics, strict evaluation, bounded sweeps, stale proposals, atomic replacement failure, protocol adapters, request truncation, CSRF, redirect refusal and managed process cleanup.
- Ruff lint/format and JavaScript syntax checks pass.
- CI runs the same Python tests on Linux, macOS and Windows with Python 3.10 and 3.12, plus a Docker image build. Inspect the repository Actions run for its current outcome.
- OpenAI-compatible and Anthropic message adapters are exercised against local protocol fixtures. No paid cloud call was made.
- Soup recipes were checked against pinned source field names. The [schema check](../validation/soup-schema-check.json) does not establish successful fine-tuning or backend compatibility.

## Live container execution

A Python unittest project was copied and executed in `python:3.12-slim-bookworm` with no network, 128 MiB memory, one CPU, read-only root filesystem and an unprivileged user. It passed with exit code 0. The image ID, command, resource limits and captured output are retained in [container evidence](../validation/docker-test.json).

The Docker application image is built separately from the project test runner. It contains the GUI/CLI and bundled static assets, not llama.cpp or model weights. Container application startup and final GUI workflow evidence are recorded in `validation/application-checks.json` when available.

## Coverage limits

Live GPU execution and memory telemetry still need validation on NVIDIA hardware. Linux and Windows CI prove software behaviour within those runners; they do not establish CUDA, Vulkan, Metal, driver or model compatibility. MLX, speculative decoding, lower-precision KV caches, external high-concurrency servers and Soup training are configurable/researched paths without live validation in this record. Router scores use configured estimates and do not yet learn automatically from production agent traffic. The benchmark workload is sequential.

The bundled tasks are not a specialised-domain benchmark. Replace them with held-out task data and evaluate accuracy, latency, memory and failure behaviour before choosing a configuration for real use. The application searches bounded configurations; it does not prove an exhaustive hardware limit.
