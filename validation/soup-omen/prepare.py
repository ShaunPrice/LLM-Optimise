import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path.cwd()
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
model_dir = snapshot_download(
    model_id,
    revision=revision,
    token=False,
    allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.txt", "*.jinja"],
    local_dir=root / "model",
)
model_files = {}
for file in sorted(Path(model_dir).glob("*")):
    if file.is_file():
        model_files[file.name] = {
            "bytes": file.stat().st_size,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
(root / "model-manifest.json").write_text(
    json.dumps({"id": model_id, "revision": revision, "files": model_files}, indent=2)
)
rows = [
    {
        "instruction": "Classify the sentiment as positive or negative. Output only the label.",
        "input": text,
        "output": label,
    }
    for text, label in [
        ("I love this tool.", "positive"),
        ("This device is terrible.", "negative"),
        ("The result is excellent.", "positive"),
        ("I hate the slow response.", "negative"),
        ("It works perfectly.", "positive"),
        ("The output is broken.", "negative"),
        ("I am delighted.", "positive"),
        ("I am disappointed.", "negative"),
    ]
]
(root / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
for mode in ["resident", "stream"]:
    config = {
        "base": str(root / "model"),
        "task": "sft",
        "backend": "transformers",
        "data": {
            "train": str(root / "train.jsonl"),
            "format": "alpaca",
            "val_split": 0.0,
            "max_length": 128,
        },
        "training": {
            "epochs": 1,
            "lr": 0.0002,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "seed": 42,
            "data_seed": 42,
            "warmup_ratio": 0.0,
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
            "quantization": "4bit",
            "gradient_checkpointing": True,
            "save_steps": 100,
            "logging_steps": 1,
        },
        "output": str(root / ("adapter-" + mode)),
    }
    if mode == "stream":
        config["training"].update(stream_layers=True, stream_source="ram", stream_buffers=2)
    (root / (mode + ".json")).write_text(json.dumps(config, indent=2))
print(
    json.dumps(
        {
            "model_id": model_id,
            "revision": revision,
            "bytes": sum(v["bytes"] for v in model_files.values()),
            "rows": len(rows),
        }
    )
)
