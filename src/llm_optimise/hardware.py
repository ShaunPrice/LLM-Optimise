"""Hardware capabilities and sampled process memory, without importing a tensor runtime."""

from __future__ import annotations

import csv
import io
import platform
import shutil
import subprocess
import threading
import time

import psutil

GIB = 1024**3


def command_output(argv, timeout=3):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def detect_hardware():
    memory = psutil.virtual_memory()
    unified = platform.system() == "Darwin" and platform.machine() == "arm64"
    gpu_text = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if gpu_text:
        for row in csv.reader(io.StringIO(gpu_text)):
            try:
                gpus.append(
                    {
                        "name": row[0].strip(),
                        "memory_gib": float(row[1]) / 1024,
                        "driver": row[2].strip(),
                    }
                )
            except (IndexError, ValueError):
                continue
    cpu = (
        command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if platform.system() == "Darwin"
        else platform.processor()
    )
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu": cpu or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "ram_total_gib": memory.total / GIB,
        "ram_available_gib": memory.available / GIB,
        "unified_memory": unified,
        "gpus": gpus,
        "python": platform.python_version(),
        "runtimes": {
            name: shutil.which(name) for name in ("llama-server", "soup", "mlx_lm.server")
        },
        "notes": ["Apple GPU shares system memory; process RSS is not Metal peak allocation."]
        if unified
        else [],
    }


def process_gpu_bytes(pids: set[int]) -> int | None:
    output = command_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]
    )
    if output is None:
        return None
    total = 0
    matched = False
    for row in csv.reader(io.StringIO(output)):
        try:
            if int(row[0].strip()) in pids:
                matched = True
                total += int(float(row[1].strip()) * 1024**2)
        except (IndexError, ValueError):
            return None
    # No process row is not proof of zero: WDDM, permissions and driver modes can hide it.
    return total if matched else None


class ResourceMonitor:
    """Cooperative sampled guard, not an OS-enforced hard memory sandbox."""

    def __init__(self, pid: int | None, limits, cpu_only=False, interval=0.05):
        self.pid, self.limits, self.cpu_only, self.interval = pid, limits, cpu_only, interval
        self.rss_peak = None
        self.gpu_peak = 0 if cpu_only and pid is not None else None
        self.swap_peak = None
        self.available_min = None
        self.samples = 0
        self.gpu_samples = 0
        self.violation = None
        self._stop = threading.Event()
        self._thread = None
        self._last_gpu = 0.0

    def start(self):
        self.sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def sample(self):
        memory = psutil.virtual_memory()
        self.available_min = min(self.available_min or memory.available, memory.available)
        if memory.available < self.limits.min_available_gib * GIB:
            self.violation = "system available RAM fell below min_available_gib"
        if self.pid is None:
            return
        try:
            root = psutil.Process(self.pid)
            processes = [root, *root.children(recursive=True)]
            rss, swap, pids = 0, 0, set()
            for process in processes:
                try:
                    info = process.memory_info()
                    rss += info.rss
                    if hasattr(info, "swap"):
                        swap += info.swap
                    pids.add(process.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if not pids:
                return
            self.rss_peak = max(self.rss_peak or 0, rss)
            self.samples += 1
            if self.limits.max_rss_gib is not None and rss > self.limits.max_rss_gib * GIB:
                self.violation = "sampled process RSS exceeded max_rss_gib"
            if not self.cpu_only and time.monotonic() - self._last_gpu >= 1:
                self._last_gpu = time.monotonic()
                gpu = process_gpu_bytes(pids)
                if gpu is not None:
                    self.gpu_peak = max(self.gpu_peak or 0, gpu)
                    self.gpu_samples += 1
                    if self.limits.max_gpu_gib is not None and gpu > self.limits.max_gpu_gib * GIB:
                        self.violation = "sampled process GPU memory exceeded max_gpu_gib"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4)
        return self.result()

    def result(self):
        return {
            "rss_peak_gib": None if self.rss_peak is None else self.rss_peak / GIB,
            "gpu_peak_gib": None if self.gpu_peak is None else self.gpu_peak / GIB,
            "system_available_min_gib": None
            if self.available_min is None
            else self.available_min / GIB,
            "samples": self.samples,
            "gpu_samples": self.gpu_samples,
            "scope": "managed process tree"
            if self.pid
            else "external endpoint; server memory unavailable",
            "gpu_source": "CPU explicitly forced"
            if self.cpu_only and self.pid
            else "NVIDIA process samples; null when unsupported",
            "violation": self.violation,
            "sample_interval_s": self.interval,
        }
