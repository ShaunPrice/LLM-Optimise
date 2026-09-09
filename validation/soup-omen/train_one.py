import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil
import torch

root = Path.cwd()
mode = sys.argv[1]
result = {
    "mode": mode,
    "status": "started",
    "gpu_allocator_limit_gib": 4.0,
    "measurement_note": "Process RSS sampled at 50 ms. CUDA allocated/reserved peaks cover this process, not whole-device usage.",
}
t0 = time.perf_counter()
peak_rss = 0
stop = threading.Event()


def watch():
    global peak_rss
    p = psutil.Process()
    while not stop.wait(0.05):
        peak_rss = max(peak_rss, p.memory_info().rss)
        if p.memory_info().rss > 8 * 1024**3:
            os._exit(125)


thread = threading.Thread(target=watch, daemon=True)
thread.start()
try:
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(
        4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    torch.cuda.reset_peak_memory_stats()
    from soup_cli.config.schema import SoupConfig
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from transformers import set_seed

    config = SoupConfig.model_validate_json((root / (mode + ".json")).read_text())
    set_seed(config.training.seed)
    wrapper = SFTTrainerWrapper(config, device="cuda", report_to="none")
    dataset = load_dataset(config.data)
    setup_start = time.perf_counter()
    wrapper.setup(dataset)
    result["setup_seconds"] = time.perf_counter() - setup_start
    # Record adapter initial state so a no-op run cannot count as passing.
    initial = {
        k: v.detach().cpu().clone() for k, v in wrapper.model.named_parameters() if v.requires_grad
    }
    result["initial_trainable_sha256"] = hashlib.sha256(
        b"".join(k.encode() + v.float().numpy().tobytes() for k, v in sorted(initial.items()))
    ).hexdigest()
    train_start = time.perf_counter()
    summary = wrapper.train()
    torch.cuda.synchronize()
    result["train_and_save_seconds"] = time.perf_counter() - train_start
    result["training_summary"] = summary
    result["global_step"] = wrapper.trainer.state.global_step
    result["log_history"] = wrapper.trainer.state.log_history
    result["changed_trainable_tensors"] = sum(
        not torch.equal(initial[k], v.detach().cpu())
        for k, v in wrapper.model.named_parameters()
        if v.requires_grad
    )
    result["trainable_tensors"] = len(initial)
    result["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    result["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    result["status"] = (
        "passed"
        if result["global_step"] == 8 and result["changed_trainable_tensors"] > 0
        else "failed"
    )
except BaseException as exc:
    result["status"] = "failed"
    result["error"] = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
finally:
    stop.set()
    thread.join()
    result["elapsed_seconds"] = time.perf_counter() - t0
    result["peak_rss_gib"] = peak_rss / 1024**3
    result["artifacts"] = {
        p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in (root / ("adapter-" + mode)).glob("*")
        if p.is_file()
    }
    (root / (mode + "-result.json")).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
sys.exit(0 if result["status"] == "passed" else 1)
