import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx

root = Path(__file__).resolve().parents[2]
assert mx.metal.is_available(), "Metal unavailable"
mx.set_memory_limit(3 * 1024**3)
mx.set_cache_limit(256 * 1024**2)
mx.random.seed(20260909)
mx.reset_peak_memory()
started = time.perf_counter()
code = 0
try:
    from soup_cli.cli import run

    sys.argv = ["soup", "train", "--config", str(root / "runs/soup-mac/soup.json"), "--yes"]
    run()
except SystemExit as exc:
    code = exc.code or 0
    raise
except BaseException:
    code = 1
    raise
finally:
    (root / "runs/soup-mac/train-process.json").write_text(
        json.dumps(
            {
                "exit_code": code,
                "elapsed_s": time.perf_counter() - started,
                "mlx_peak_memory_bytes": mx.get_peak_memory(),
                "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "mlx_memory_limit_bytes": 3 * 1024**3,
                "mlx_cache_limit_bytes": 256 * 1024**2,
                "versions": {
                    x: importlib.metadata.version(x)
                    for x in ["mlx", "mlx-lm", "soup-cli", "transformers", "psutil"]
                },
                "python": sys.version,
            },
            indent=2,
        )
        + "\n"
    )
