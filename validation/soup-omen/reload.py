import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

root = Path.cwd()
mode = sys.argv[1]
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(
    4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
)
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(
    root / "model", local_files_only=True, trust_remote_code=False
)
base = AutoModelForCausalLM.from_pretrained(
    root / "model",
    local_files_only=True,
    trust_remote_code=False,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
)
model = PeftModel.from_pretrained(base, root / ("adapter-" + mode), is_trainable=False)
model.eval()
disk = load_file(str(root / ("adapter-" + mode) / "adapter_model.safetensors"))
loaded = get_peft_model_state_dict(model)
missing_keys = sorted(set(disk) - set(loaded))
max_weight_error = max(
    float((disk[k].float() - loaded[k].detach().cpu().float()).abs().max())
    for k in disk
    if k in loaded
)
nonzero_b = sum(bool(torch.count_nonzero(v)) for k, v in loaded.items() if "lora_B" in k)
records = []
for text, label in [
    ("I really enjoyed the result.", "positive"),
    ("This result is awful.", "negative"),
]:
    prompt = f"### Instruction:\nClassify the sentiment as positive or negative. Output only the label.\n\n### Input:\n{text}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        with model.disable_adapter():
            base_logits = model(**inputs).logits.float()
        adapted_logits = model(**inputs).logits.float()
        delta = float((base_logits - adapted_logits).abs().max())
        tokens = model.generate(
            **inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
        output = tokenizer.decode(tokens[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    records.append(
        {
            "input": text,
            "expected": label,
            "output": output,
            "base_adapter_max_logit_difference": delta,
            "finite_logits": bool(torch.isfinite(adapted_logits).all()),
            "generated_tokens": int(tokens.shape[1] - inputs.input_ids.shape[1]),
        }
    )
result = {
    "mode": mode,
    "missing_adapter_keys": missing_keys,
    "saved_loaded_max_weight_error": max_weight_error,
    "loaded_nonzero_lora_B_tensors": nonzero_b,
    "saved_tensor_count": len(disk),
    "heldout_smoke": records,
    "elapsed_seconds": time.perf_counter() - start,
    "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
}
result["status"] = (
    "passed"
    if not missing_keys
    and max_weight_error == 0
    and nonzero_b > 0
    and all(
        r["finite_logits"]
        and r["base_adapter_max_logit_difference"] > 1e-7
        and r["generated_tokens"] > 0
        for r in records
    )
    else "failed"
)
(root / (mode + "-reload.json")).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
sys.exit(0 if result["status"] == "passed" else 1)
