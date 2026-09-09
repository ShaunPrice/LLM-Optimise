import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

root = Path(__file__).resolve().parents[2]
out = root / "runs/soup-mac"
env = dict(
    os.environ,
    SOUP_DB_PATH=str(out / "experiments.db"),
    HF_HOME=str(root / ".work/soup-mac/hf-cache"),
    HF_HUB_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    TOKENIZERS_PARALLELISM="false",
)
start = time.perf_counter()
peak = 0
minimum_available = psutil.virtual_memory().available
reason = None
with (out / "train.log").open("w") as log:
    child = subprocess.Popen(
        [
            str(root / ".work/soup-mac/venv/bin/python"),
            str(root / "validation/soup-mac/train_entry.py"),
        ],
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    while child.poll() is None:
        try:
            proc = psutil.Process(child.pid)
            rss = sum(
                p.memory_info().rss
                for p in [proc, *proc.children(recursive=True)]
                if p.is_running()
            )
            peak = max(peak, rss)
            minimum_available = min(minimum_available, psutil.virtual_memory().available)
            if rss > 4 * 1024**3:
                reason = "4GiB process RSS limit"
                child.terminate()
            if time.perf_counter() - start > 300:
                reason = "300s timeout"
                child.terminate()
            if reason:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.05)
result = {
    "exit_code": child.returncode,
    "elapsed_s": time.perf_counter() - start,
    "sampled_peak_rss_bytes": peak,
    "minimum_host_available_bytes": minimum_available,
    "sampling_interval_s": 0.05,
    "termination_reason": reason,
}
(out / "monitor.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
sys.exit(child.returncode)
