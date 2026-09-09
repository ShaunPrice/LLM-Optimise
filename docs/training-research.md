# Research and integration decisions

Reviewed 9 September 2026. This document separates implemented controls, integrations with external runtimes, and research paths. Performance depends on the task, model, runtime build and hardware; published speedups are not results from this application.

## Soup: the project requested

The requested [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup) is a fine-tuning toolkit. It is distinct from the model-weight averaging technique called *model soups*. We inspected revision [`254351e`](https://github.com/MakazhanAlpamys/Soup/tree/254351e10331e360b5d75e26aabe0a4e591e16a4).

Soup's layer streaming moves frozen base layers between host storage/memory and the accelerator during adapter training. It can make larger training jobs feasible with small VRAM, while transfer bandwidth and host RAM become constraints. LLM-Optimise emits explicit batch-one, four-bit, LoRA recipes for streaming, resident QLoRA and MLX. Streaming buffers and source are recorded in the configuration. [Soup performance guidance](https://github.com/MakazhanAlpamys/Soup/blob/254351e10331e360b5d75e26aabe0a4e591e16a4/docs/performance-and-quantization.md)

The three generated recipes were checked against field names in Soup's configuration source. That initial check was an AST/schema-field check, not training. Subsequent live MLX and CUDA training/reload tests are documented in [validation](validation.md). `data.max_length` belongs under `data`; streaming options belong under `training`. [Pinned schema](https://github.com/MakazhanAlpamys/Soup/blob/254351e10331e360b5d75e26aabe0a4e591e16a4/src/soup_cli/config/schema.py), [check artifact](../validation/soup-schema-check.json)

MLX and layer streaming are separate routes. Use MLX on Apple silicon with compatible models/data; do not combine it with the Transformers streaming switches. Upstream documents backend-specific limits, including MLX adapter/resume behaviour and experimental MPS support. [Soup backend guidance](https://github.com/MakazhanAlpamys/Soup/blob/254351e10331e360b5d75e26aabe0a4e591e16a4/docs/backends-and-ops.md)

Soup is evolving rapidly. Upstream release notes include adapter-loading and training-correctness fixes. Pin a reviewed version, retain a resident control, reload the saved adapter into a fresh process, and compare held-out quality before treating a training run as successful. No upstream throughput claim is presented here as a result from our machine. [Soup release history](https://github.com/MakazhanAlpamys/Soup/releases)

## Technology choices

| Technology | What it can change | Application status and useful experiment |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / GGUF quantisation | Weight size, CPU/GPU execution, accuracy | **Implemented managed runtime.** Compare model files, threads, context, GPU layers, batches, mmap and cache types with identical tasks. |
| [llama.cpp server controls](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | KV cache, Flash Attention, prompt reuse, draft-model speculation, structured output | **Implemented configuration controls.** Only schema constraints, CPU and Metal paths were exercised in the recorded live validation. Runtime support varies. |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | Apple unified-memory inference and adapter training | **External runtime path.** Compatible HTTP endpoints can be registered; Soup MLX recipes can be exported. No bundled native MLX engine. |
| [QLoRA](https://arxiv.org/abs/2305.14314) | Adapter training with a quantised frozen base | **Recipe integration.** Compare resident QLoRA with Soup streaming before increasing model size. Bounded resident/streaming CUDA training and adapter reload now pass on Omen; this is not held-out task-quality validation. |
| [vLLM](https://github.com/vllm-project/vllm) | Batched serving and KV memory management | **External compatible endpoint.** Useful for concurrent serving; this lab currently measures sequential request latency, not a load-testing throughput ceiling. |
| [SGLang](https://github.com/sgl-project/sglang) | Serving and prefix reuse | **External compatible endpoint.** Calibrate under the real workload; external process RAM/VRAM is not observed by this lab. |
| [BitNet](https://github.com/microsoft/BitNet) | CPU inference for supported low-bit architectures | **Research path.** Requires compatible models/runtime; it is not a generic lossless conversion for arbitrary GGUF files. No adapter implemented. |
| [Model soups](https://arxiv.org/abs/2203.05482) | Averaging compatible fine-tuned weights | **Research path.** Primarily a quality/generalisation technique; it does not inherently shrink parameter count. No averaging implementation. |

## How to push constrained hardware usefully

1. Define held-out examples for the specialised task and a minimum acceptable score. Keep prompt selection/training data separate.
2. Measure one baseline in a fresh managed process. Record hashes, runtime revision, context, completion limits, warmups and repeats.
3. Sweep one small set of hypotheses: smaller quantisation, shorter context, CPU/GPU offload, KV precision or schema-constrained output. Retain failed trials.
4. Compare quality, latency and measured memory together. A fitting model that fails the task is not a viable result.
5. Use Pareto candidates to guide the next sweep. Memory estimation helps plan it; sampling establishes observations. Neither is a proof that all configurations have been searched.
6. If the task still needs improvement, prepare an adapter-training experiment, execute it in a separate pinned environment, reload the adapter, and evaluate on the held-out set.
7. Calibrate the routing catalogue from representative agent tasks. Declared costs, quality and memory are estimates until backed by measurements; avoid transferring a six-task smoke score to unrelated code generation.

The first live validation demonstrates an application workflow and one useful structured-output improvement. It does not establish the limits of an M4, NVIDIA GPU, or any model family. See [validation evidence](validation.md).


## Experimental workbench additions

The [Workbench guide](workbench.md) documents executable dataset, adapter, search, calibration, caching, context, distillation, component, repair and model-lifecycle workflows shared by the GUI and CLI. Native process supervision and desktop packaging are documented under [native](../native/README.md) and [desktop](../desktop/README.md).
