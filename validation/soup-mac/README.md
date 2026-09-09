# Soup on Apple M4: executed training and adapter reload

**Passed on 9 September 2026.** Soup's actual `soup train` CLI completed 12 MLX supervised fine-tuning iterations, wrote a LoRA adapter, and a separate process reloaded it and generated an answer. This is a small integration smoke test, not a task benchmark or the limit of the machine.

| Observation | Measured result |
|---|---:|
| Machine | Apple M4, 24 GiB unified memory |
| Soup | 0.74.0 at `254351e10331e360b5d75e26aabe0a4e591e16a4` |
| Runtime | Python 3.12.13; MLX 0.32.2; mlx-lm 0.31.3; Transformers 5.16.1 |
| Model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, pinned revision below |
| Training | 12 rows, 12 iterations, batch 1, accumulation 1, LoRA rank 4 / alpha 8, context 128 |
| Supervision | Assistant responses only; gradient checkpointing enabled |
| Training-loop duration, including display/tracker | 8.061 seconds |
| Whole process, including imports/loading/CLI | 13.327 seconds externally timed |
| Training loss, first / last reported batch | 1.1002 / 0.1533 |
| MLX allocator high-water mark | 416,653,170 bytes = **0.388 GiB** |
| Sampled process-tree RSS | 558,923,776 bytes = **0.521 GiB** |
| OS process RSS high-water mark | 569,573,376 bytes = **0.530 GiB** |
| Adapter file | 1,091,411 bytes; 96 tensors |
| Fresh-process reload | All 96 saved tensors attached exactly; 48 nonzero LoRA B tensors |
| Model effect | Maximum absolute next-token logit difference: 7.89453125 |

MLX's console prints **decimal GB**, despite some upstream prose using GiB. The JSON artifact stores bytes. Allocator usage and RSS overlap in Apple unified memory; they must not be added. The 50 ms RSS sampling interval can miss transients, so the OS high-water mark is included separately. The reported loss values concern different tiny training batches and do not measure held-out accuracy.

For the single untrained illustration `temperature=70`, the base model generated `alarm`; the reloaded adapter generated `normal`, matching the synthetic rule. One example is evidence of an exercised inference path, not a quality score or a generalisation result. No real industrial data was used.

## Evidence

- [Structured result](result.json): hardware, pinned revisions, hashes, effective config, step metrics, timing, memory and reload assertions.
- [Training log](train.log) and [reload/inference log](reload.log): real executions with workspace paths redacted.
- [Synthetic training rows](train.jsonl): 12 fictional temperature classifications; the inference value 70 is absent.
- [Dependency lock](requirements-lock.txt): exact installed distribution versions and pinned Soup source.

The model's revision is `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`. Downloaded model/tokenizer files total 289,598,797 bytes. Model weights, adapters, virtual environments, caches and the private tracker database remain in ignored local directories; they are not committed.

## Reproduce on Apple silicon

From the repository root, with Python 3.12 and `uv` installed:

```sh
mkdir -p .work/soup-mac runs/soup-mac
UV_CACHE_DIR=.work/soup-mac/uv-cache uv venv --python python3.12 .work/soup-mac/venv
UV_CACHE_DIR=.work/soup-mac/uv-cache uv pip install --python .work/soup-mac/venv/bin/python -r validation/soup-mac/requirements-lock.txt
HF_HOME=.work/soup-mac/hf-cache HF_HUB_DISABLE_TELEMETRY=1 .work/soup-mac/venv/bin/python validation/soup-mac/download_model.py
.work/soup-mac/venv/bin/python validation/soup-mac/prepare.py
.work/soup-mac/venv/bin/python validation/soup-mac/monitor.py
HF_HOME=.work/soup-mac/hf-cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false .work/soup-mac/venv/bin/python validation/soup-mac/reload.py
```

These commands write/replace the local `runs/soup-mac/` outputs. The preparation script uses LLM-Optimise's recipe builder, then explicitly sets a zero validation split, accumulation 1, logging every step and saving at step 12 for this bounded fixture. Soup's real config validation and Alpaca dataset conversion run through its CLI.

The monitor invokes the equivalent of:

```sh
SOUP_DB_PATH=runs/soup-mac/experiments.db HF_HUB_OFFLINE=1 .work/soup-mac/venv/bin/soup train --config runs/soup-mac/soup.json --yes
```

The Python entry adds MLX allocator/cache limits of 3 GiB / 256 MiB, an explicit `mx.random` seed, and allocator/OS high-water reporting. The external monitor samples RSS every 50 ms, terminates its own training process if RSS exceeds 4 GiB, and sets a 300-second timeout. These are cooperative experiment controls, not an OS sandbox or a total-machine memory guarantee. The test reached neither stop condition. Soup displays an unset seed default of 42; that display does not establish MLX seeding, so the entry explicitly seeds MLX with `20260909`. Cross-run determinism was not tested.

The reload program starts after training has exited. It checks every saved tensor exists in the loaded model and equals the saved value; checks finite values and nonzero LoRA B updates; checks baseline/adapter logits differ; and runs greedy generation capped at 12 tokens. It does not execute generated text.

## Scope and upstream sources

This run validates **resident quantised MLX LoRA training on Mac**. It does not exercise layer streaming, CUDA, MPS training, model export, resumable optimiser state, or the maximum trainable model size. MLX and Soup layer streaming are separate backends. No upstream files were patched for this successful run.

- [Pinned Soup MLX trainer](https://github.com/MakazhanAlpamys/Soup/blob/254351e10331e360b5d75e26aabe0a4e591e16a4/src/soup_cli/trainer/mlx_sft.py)
- [Pinned backend guidance](https://github.com/MakazhanAlpamys/Soup/blob/254351e10331e360b5d75e26aabe0a4e591e16a4/docs/backends-and-ops.md)
- [Upstream's small-model MLX fixture](https://github.com/MakazhanAlpamys/Soup/blob/254351e10331e360b5d75e26aabe0a4e591e16a4/benchmarks/harness/mlx_sft_smoke.py)
- [Pinned test model](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3)
