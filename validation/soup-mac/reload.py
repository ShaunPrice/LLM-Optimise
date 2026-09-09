import gc
import hashlib
import json
import resource
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

root = Path(__file__).resolve().parents[2]
out = root / "runs/soup-mac"
mx.set_memory_limit(3 * 1024**3)
mx.set_cache_limit(256 * 1024**2)
started = time.perf_counter()
mx.reset_peak_memory()
base, tokenizer = load(str(root / ".work/soup-mac/model"))
prompt = tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": "Classify this fictional sensor reading. Reply only normal if temperature < 80, otherwise alarm.\n\ntemperature=70",
        }
    ],
    tokenize=False,
    add_generation_prompt=True,
)
tokens = mx.array([tokenizer.encode(prompt)])
base_logits = base(tokens)[:, -1, :].astype(mx.float32)
mx.eval(base_logits)
base_text = generate(
    base, tokenizer, prompt=prompt, max_tokens=12, sampler=make_sampler(temp=0.0), verbose=False
)
del base
gc.collect()
mx.clear_cache()
model, tokenizer = load(str(root / ".work/soup-mac/model"), adapter_path=str(out / "adapter"))
saved = mx.load(str(out / "adapter/adapters.safetensors"))
loaded = dict(tree_flatten(model.parameters()))
missing = [key for key in saved if key not in loaded]
assert not missing, missing
exact = all(bool(mx.array_equal(value, loaded[key]).item()) for key, value in saved.items())
assert exact, "Saved tensors differ from attached adapter"
nonzero = [
    key
    for key, value in saved.items()
    if key.endswith("lora_b") and bool(mx.any(value != 0).item())
]
assert nonzero, "No updated LoRA B tensors"
assert all(bool(mx.all(mx.isfinite(x)).item()) for x in saved.values())
adapted_logits = model(tokens)[:, -1, :].astype(mx.float32)
mx.eval(adapted_logits)
diff = float(mx.max(mx.abs(adapted_logits - base_logits)).item())
assert diff > 0, "Adapter has no observed effect on logits"
adapted_text = generate(
    model, tokenizer, prompt=prompt, max_tokens=12, sampler=make_sampler(temp=0.0), verbose=False
)
assert adapted_text.strip(), "Empty generation"
result = {
    "status": "passed",
    "fresh_process_reload": True,
    "adapter_tensor_count": len(saved),
    "all_saved_tensors_attached_exactly": exact,
    "nonzero_lora_b_tensors": len(nonzero),
    "all_adapter_values_finite": True,
    "baseline_vs_adapter_max_abs_logit_difference": diff,
    "inference_prompt": prompt,
    "baseline_generation": base_text,
    "adapted_generation": adapted_text,
    "expected_for_this_example": "normal",
    "max_generated_tokens": 12,
    "sampling_temperature": 0,
    "elapsed_s": time.perf_counter() - started,
    "mlx_peak_memory_bytes": mx.get_peak_memory(),
    "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "adapter_bytes": (out / "adapter/adapters.safetensors").stat().st_size,
    "adapter_sha256": hashlib.sha256(
        (out / "adapter/adapters.safetensors").read_bytes()
    ).hexdigest(),
}
(out / "reload.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
