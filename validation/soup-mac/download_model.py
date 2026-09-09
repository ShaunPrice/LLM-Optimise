"""Fetch only the public model files used by the recorded test, at its exact revision."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / "validation/soup-mac/result.json").read_text())["model"]
assert manifest["download_bytes"] < 1024**3
path = snapshot_download(
    manifest["model_id"],
    revision=manifest["revision"],
    local_dir=root / ".work/soup-mac/model",
    allow_patterns=[entry["path"] for entry in manifest["files"]],
    token=False,
    max_workers=3,
)
print(path)
