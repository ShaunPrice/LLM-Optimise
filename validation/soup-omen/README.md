# Soup on Omen: live CUDA training validation

Tested **9 September 2026** on Omen's existing Ubuntu 24.04 WSL2 environment, hosted by Windows 11 Pro. Hardware: Intel Core Ultra 9 275HX, approximately 64 GiB host RAM, NVIDIA RTX 5090 Laptop with 24,463 MiB device memory, driver 610.62. This is WSL/Linux CUDA validation; native Windows training was not exercised.

Both resident NF4 QLoRA and RAM layer-streamed NF4 completed real training and saved usable adapters. This is an eight-example, eight-step **training and adapter smoke test**, not a successful specialised-task benchmark or a hardware-limit measurement.

## Recorded results

The final check ran streaming first, then resident, after both paths had already completed once. Values below come from the final check, with seed 42 explicitly set before model setup and in Soup's configuration.

| Mode | Setup | Train and save | Sampled process RAM peak | PyTorch CUDA allocated peak | PyTorch CUDA reserved peak |
|---|---:|---:|---:|---:|---:|
| Resident NF4 | 4.674 s | 7.399 s | 2.134 GiB | 0.280 GiB | 0.313 GiB |
| RAM layer-streamed NF4 | 6.489 s | 7.166 s | 2.202 GiB | 0.219 GiB | 0.221 GiB |

The first pass included cold setup work: resident setup/train-and-save took 20.526/8.470 seconds; streaming took 9.319/5.854 seconds. The complete earlier measurements are retained in [first-pass.json](first-pass.json). Setup, imports, trainer-internal runtime and wrapper train-and-save timing are different measurement intervals; the result files preserve each interval.

These are short sequential observations. Initial trainable-parameter fingerprints differ across the two implementations; those fingerprints include parameter names, so numerical initialisation parity was not established. No general speedup, equivalent training trajectory, or quality advantage follows from this table.

## What passed

- Each mode completed eight optimiser steps with finite recorded losses; all 120 trainable parameter tensors changed.
- Each final adapter was reloaded in a new Python process using PEFT. All 120 saved adapter tensors matched the loaded values exactly, with no missing keys.
- Each loaded adapter contained 60 nonzero LoRA-B tensors and changed model logits compared with the same model with its adapter disabled.
- Both adapters generated finite outputs on two held-out sentiment prompts.
- **Exact task-format quality did not pass:** the model returned explanatory sentences instead of only `positive` or `negative`. Do not count successful training or semantically plausible text as a passed application quality gate.

The harness calls Soup's real `SoupConfig`, `load_dataset`, `SFTTrainerWrapper.setup()` and `train()` APIs. It is not a replacement optimiser and does not merely check configuration fields. The generated JSON configs can also be used with `soup train --config`; this test's measured run used the instrumented Python API so it could inspect weights and allocator peaks directly.

## Scope and resource limits

The test used `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, one epoch, eight public synthetic rows, batch one, accumulation one, maximum length 128, rank four LoRA on query/value projections, and NF4 base quantisation. Soup was pinned to [`254351e`](https://github.com/MakazhanAlpamys/Soup/tree/254351e10331e360b5d75e26aabe0a4e591e16a4), version 0.74.0. Exact package versions and model checksums are included.

The **4 GiB limit applies to the process's PyTorch CUDA allocator**, not total device VRAM. Driver/context memory, other processes and allocations outside that allocator are excluded. Process RSS was sampled every 50 ms, so short peaks may be missed. [gpu-after.csv](gpu-after.csv) is only an after-test device snapshot; it is not a peak measurement. Omen's unrelated workloads were left running.

Dependencies, downloaded model and adapters remain in Omen's dedicated `/home/nexus/llm-optimise-soup-validation` workspace. Soup initially wrote this test model's shard cache to its default user cache; that exact cache was relocated into the dedicated workspace afterwards. `SOUP_LAYER_STREAM_CACHE_DIR` points there for future runs. No driver, global Python environment, unrelated project or service was modified.

## Reproduce

Copy the Python scripts and `requirements-observed.txt` from this directory into an empty working directory in an existing CUDA-capable Ubuntu/WSL environment. The recorded environment used Python 3.12.3 and PyTorch 2.11.0+cu128. Install through a virtual environment; the CUDA wheel download and model download require network access.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-observed.txt
export HF_HOME="$PWD/hf-cache"
export SOUP_LAYER_STREAM_CACHE_DIR="$PWD/soup-layer-cache"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 WANDB_MODE=disabled
.venv/bin/python prepare.py
export HF_HUB_OFFLINE=1
timeout 300 .venv/bin/python train_one.py resident > resident.log 2>&1
timeout 300 .venv/bin/python train_one.py stream > stream.log 2>&1
timeout 120 .venv/bin/python reload.py resident > resident-reload.log 2>&1
timeout 120 .venv/bin/python reload.py stream > stream-reload.log 2>&1
```

`prepare.py` downloads the pinned 135M model and writes the synthetic dataset and exact configs. It overwrites files with those names; use a dedicated empty directory. For a fresh comparison, control cache state, run order, repeat count and initial adapter tensors explicitly. The commands above reproduce the test procedure, not a timing guarantee.

## Evidence

- [Machine and scope summary](summary.json), [package environment](environment.json), [observed dependencies](requirements-observed.txt), [model hashes](model-manifest.json).
- [Resident training](resident-result.json), [resident reload](resident-reload.json), [resident log](resident.log).
- [Streaming training](stream-result.json), [streaming reload](stream-reload.json), [streaming log](stream.log).
- [Resident recipe](resident.json), [streaming recipe](stream.json), [synthetic dataset](train.jsonl), [training harness](train_one.py), [reload harness](reload.py).

No credentials, model weights, binary adapters or private task data are published in this evidence directory.
