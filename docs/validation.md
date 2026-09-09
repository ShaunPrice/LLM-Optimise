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

- 150 local automated tests pass, including routing constraints, unknown metrics, strict evaluation, bounded sweeps, stale proposals, atomic replacement failure, protocol adapters, request truncation, CSRF, redirect refusal and managed process cleanup.
- Ruff lint/format and JavaScript syntax checks pass.
- All seven [CI jobs passed](https://github.com/ShaunPrice/LLM-Optimise/actions/runs/34340038933) at source commit `1e40e5d`: Python 3.10/3.12 on Linux, macOS and Windows, plus Docker. [Recorded CI status](../validation/ci.json).
- All seven jobs passed again after the OpenRouter/report-loader changes at source commit `2c928c9`, including the 150-test suite on each Python/OS combination. [Latest CI run](https://github.com/ShaunPrice/LLM-Optimise/actions/runs/34342800572), [recorded result](../validation/ci-cloud-soup.json).
- OpenRouter cloud requests and GUI chat now have live validation; direct Anthropic requests still have local protocol-fixture coverage only. See the added cloud evidence below.
- Soup recipes were checked against pinned source field names. The [schema check](../validation/soup-schema-check.json) does not establish successful fine-tuning or backend compatibility.

## Live container execution

A Python unittest project was copied and executed in `python:3.12-slim-bookworm` with no network, 128 MiB memory, one CPU, read-only root filesystem and an unprivileged user. It passed with exit code 0. The image ID, command, resource limits and captured output are retained in [container evidence](../validation/docker-test.json).

The Docker application image is built separately from the project test runner. It contains the GUI/CLI and bundled static assets, not llama.cpp or model weights. The refreshed Docker image served the API and PNG icon successfully with 256 MiB RAM and one CPU. [Application checks](../validation/application-checks.json) record its image ID and the live GUI workflow: local 1.5B model generation, diff review, apply, four passing container tests, local chat and routing preview. A separate [timeout check](../validation/docker-timeout.json) stopped an overlong container with exit 137.

The image was rebuilt after the OpenRouter/report-loader changes and again served its API and PNG icon with the same 256 MiB/one-CPU read-only profile. [Refreshed container check](../validation/openrouter/application-container.json).

## Coverage limits

Soup MLX training on Mac and CUDA training through WSL on Omen now have live validation. Native Windows training, llama.cpp NVIDIA inference/telemetry, speculative decoding, lower-precision KV caches and external high-concurrency servers remain unvalidated. Linux and Windows CI establish software behaviour in those runners, not accelerator or driver compatibility. Router scores use configured estimates and do not yet learn automatically from production agent traffic. The benchmark workload is sequential.

The bundled tasks are not a specialised-domain benchmark. Replace them with held-out task data and evaluate accuracy, latency, memory and failure behaviour before choosing a configuration for real use. The application searches bounded configurations; it does not prove an exhaustive hardware limit.

## Application overhead

With its model stopped and the browser polling, the GUI server process sampled 22.84 MiB RSS and 0.5% CPU over two seconds on this Mac. Browser and operating-system memory are excluded. This is a short observation, not a peak-memory bound. The interface was checked at its default desktop size and 390-pixel viewport without page overflow; the browser console reported no errors after the final reload.

## OpenRouter cloud validation

Fifteen live requests passed: six synthetic task checks each on GPT-4.1 nano and mini, one structured code proposal, one direct chat and one GUI chat. Both models scored 6/6 on the smoke tasks. The reviewed cloud-generated code passed three unittest methods covering four cases in a 128 MiB network-disabled Docker container. Total provider-reported cost was **US$0.0013586**. These are integration checks, not a cloud model ranking. [Cloud results and reproduction](../validation/openrouter/README.md).

## Soup live training

Soup 0.74.0 at pinned revision `254351e` was installed in separate environments on both machines. Both tests save adapters and reload them in fresh processes; model weights and adapters remain outside Git.

- **Mac / MLX:** 0.5B four-bit Qwen model, 12 steps, 8.061 seconds training-loop time, 0.388 GiB MLX allocation peak and 0.530 GiB OS process RSS high-water. All 96 saved adapter tensors attached correctly. [Mac evidence](../validation/soup-mac/README.md).
- **Omen / CUDA via WSL:** RTX 5090 Laptop, 24 GiB VRAM; resident and streaming four-bit 135M-model tests both completed eight optimiser steps. Fresh reloads matched all 120 saved tensors in each adapter. Timing and memory definitions are in the [Omen evidence](../validation/soup-omen/README.md).

Neither tiny training exercise establishes useful domain quality. The Omen model remained verbose on the label task, even though training and adapter updates worked. Layer streaming was tested on Omen; the Mac uses resident MLX LoRA. The GUI still exports recipes; these validation scripts execute the training separately. Proposed in-app training orchestration and Rust work are described in the [performance roadmap](performance-roadmap.md).

## Implemented workbench validation — 9 September 2026

The 16 roadmap additions now have executable paths shared by the GUI and CLI. The integrated local Python suite passes **316 tests**, including real fixture processes, budget/cache/contract tests, grouped datasets, isolated training worker controls, lifecycle admission, cancellation and native process integration. Ruff checks and formatting pass; both JavaScript files pass syntax checks. The wheel builds, and the Docker application successfully runs `workbench status` with networking disabled and a 128 MiB container limit. These software checks do not establish every accelerator or domain workload.

| Live check | Observed result | Evidence |
|---|---|---|
| Mac accelerator explorer | Same Qwen2.5-Coder-1.5B Q4 model and six smoke tasks: CPU p50 256 ms, Metal p50 120 ms; both task quality 1.0 in this single-repeat run. Sampled RSS 1.98/1.08 GiB. | [CPU/Metal](../validation/workbench/mac-accelerators.json) |
| Mac KV/context explorer | 512/1024 contexts and f16/q8_0 caches ran and passed the six smoke tasks. Reports retain the measured frontier and incomparable GPU telemetry. | [KV observations](../validation/workbench/mac-kv.json) |
| Mac managed Rust inference | Loaded a real GGUF under Rust ownership, generated YES, returned an exact cache hit, unloaded and generated NO after a fresh worker reload. Single cache lookup ~0.52 ms versus original request ~77.6 ms; not an average cache benchmark. | [Managed model/cache/reload](../validation/workbench/mac-managed-native.json) |
| Mac Soup workbench API | Twelve MLX microsteps, saved/registered adapter, fresh reload, baseline evaluation and regression comparison. All 96 saved tensors attached exactly; 48 LoRA-B tensors nonzero. Peak sampled RSS 0.55 GiB. One unseen illustration passed after adaptation, which is insufficient for domain-quality claims. | [Training API](../validation/workbench/mac-training-api.json) |
| Docker components and repair | Python/Rust each passed three host-scored component cases. Python and Rust repair failed first then passed second; original projects unchanged. Node JUnit passed. Fake printed test counts were rejected. Timeout failures remained in results alongside subsequent successes. Model responses in these repair tests were controlled fixtures. | [Component/repair evidence](../validation/workbench/docker-components-repair.json) |
| Rust supervisor on Mac | Three repeats of equal process workloads. Median idle supervisor RSS ~7.83 MiB Rust versus ~18.17 MiB Python; median startup ~4.74/28.43 ms. Cancellation ~44.94/47.77 ms, similar in this sample. | [Full profile](../native/validation/macos-arm64-supervisor-profile.json) |
| Rust supervisor on native Omen Windows | Built with the installed Rust/MSVC toolchain; job assignment, exit42, RSS termination, deadline, child cancellation and orphan cleanup passed. | [Windows native proof](../native/validation/windows-x64-supervisor-validation.json) |
| Tauri on Mac | Native webview, owned Python service HTML/state, shutdown cleanup and mounted DMG layout checked. Development bundle is unsigned for distribution and not notarized. | [Desktop validation](../desktop/README.md#validation) |

The GUI dataset workflow and [bounded real CPU capacity search](../validation/workbench/mac-gui-capacity.json) were exercised through browser controls. The new workbench exposes 25 guided tools in six areas, shares job progress/cancellation with the main application, and retains request/evidence inspection.

Additional accelerator inference paths, real speculative-decoding benefits, student quality after distillation, native Windows Soup training, and signed production installer acceptance remain unvalidated. Prior Omen Soup tests use WSL/CUDA. The native supervisor memory comparison excludes model memory, the Python API, browser/webview helpers and GPU allocations. Sampled limits can miss short-lived spikes; Metal allocation remains unknown.
