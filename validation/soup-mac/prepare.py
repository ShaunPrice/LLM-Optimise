import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
from llm_optimise.training import training_recipe  # noqa: E402

out = root / "runs/soup-mac"
out.mkdir(parents=True, exist_ok=True)
rows = []
for value, label in [
    (10, "normal"),
    (15, "normal"),
    (20, "normal"),
    (25, "normal"),
    (30, "normal"),
    (35, "normal"),
    (85, "alarm"),
    (90, "alarm"),
    (95, "alarm"),
    (100, "alarm"),
    (105, "alarm"),
    (110, "alarm"),
]:
    rows.append(
        {
            "instruction": "Classify this fictional sensor reading. Reply only normal if temperature < 80, otherwise alarm.",
            "input": f"temperature={value}",
            "output": label,
        }
    )
(out / "train.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
recipe = training_recipe(
    "soup-mlx",
    str(root / ".work/soup-mac/model"),
    str(out / "train.jsonl"),
    str(out / "adapter"),
    max_length=128,
    rank=4,
)
config = recipe["config"]
config["data"]["val_split"] = 0
config["training"].update(gradient_accumulation_steps=1, logging_steps=1, save_steps=12)
config["experiment_name"] = "m4-mlx-sft-smoke"
(out / "soup.json").write_text(json.dumps(config, indent=2) + "\n")
print(json.dumps(config, indent=2))
