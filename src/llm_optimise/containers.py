"""Resource-bounded Docker builds/tests in disposable copies of code projects."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .config import positive

IMAGES = {
    "python": "python:3.12-slim-bookworm",
    "node": "node:22-slim",
    "rust": "rust:1-slim-bookworm",
}
COMMANDS = {
    ("python", "test"): ["python", "-m", "unittest", "discover", "-s", "tests", "-v"],
    ("python", "build"): ["python", "-m", "compileall", "-q", "."],
    ("node", "test"): ["node", "--test"],
    ("node", "build"): ["npm", "run", "build"],
    ("rust", "test"): ["cargo", "test", "--offline"],
    ("rust", "build"): ["cargo", "build", "--release", "--offline"],
}
EXCLUDED = {"node_modules", "models", "runs", "__pycache__", "dist", "build"}


def copy_project(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise ValueError("project directory does not exist")
    if destination.is_relative_to(source):
        raise ValueError("container artifact directory must be outside the source project")
    files, size = [], 0
    for folder, dirs, names in os.walk(source, followlinks=False):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in EXCLUDED and not (Path(folder) / d).is_symlink()
        ]
        for name in names:
            path = Path(folder) / name
            if name.startswith(".") or path.is_symlink():
                continue
            size += path.stat().st_size
            if size > 256 * 1024**2 or len(files) >= 10000:
                raise ValueError("container context exceeds 256 MiB or 10000 files")
            files.append(path)
    destination.mkdir(parents=True, exist_ok=False)
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return len(files)


def container_command(image, project, name, command, memory_mib, cpus, network):
    # Structured argv only. User code runs after IMAGE, with no Docker socket or host privileges.
    user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000"
    return [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        name,
        "--network",
        "bridge" if network else "none",
        "--memory",
        f"{memory_mib}m",
        "--memory-swap",
        f"{memory_mib}m",
        "--cpus",
        str(cpus),
        "--pids-limit",
        "128",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=64m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        user,
        "--mount",
        f"type=bind,src={project},dst=/workspace",
        "--workdir",
        "/workspace",
        "--env",
        "HOME=/tmp",
        "--env",
        "CARGO_HOME=/tmp/cargo",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        image,
        *command,
    ]


def run_container(
    project,
    output_dir,
    *,
    runtime="python",
    action="test",
    memory_mib=512,
    cpus=1,
    timeout_s=60,
    network=False,
    pull=False,
    command=None,
    image=None,
    cancel=None,
):
    if runtime not in IMAGES or action not in ("test", "build"):
        raise ValueError("choose python/node/rust and test/build")
    positive(memory_mib, "memory_mib", integer=True)
    positive(cpus, "cpus")
    positive(timeout_s, "timeout_s")
    if memory_mib < 64 or memory_mib > 65536 or cpus > 64 or timeout_s > 3600:
        raise ValueError("limits: memory 64–65536 MiB, CPUs <=64, timeout <=3600s")
    if type(network) is not bool or type(pull) is not bool:
        raise ValueError("network and pull must be booleans")
    if not shutil.which("docker"):
        raise RuntimeError(
            "Docker CLI unavailable. Install Docker Engine/Desktop and run this command on the host."
        )
    image = image or IMAGES[runtime]
    if (
        not isinstance(image, str)
        or not image
        or image.startswith("-")
        or any(c.isspace() for c in image)
    ):
        raise ValueError("invalid container image reference")
    command = command or COMMANDS[runtime, action]
    if (
        not isinstance(command, list)
        or not command
        or len(command) > 64
        or any(not isinstance(v, str) or "\x00" in v for v in command)
    ):
        raise ValueError("container command must be a nonempty string argv list")
    inspect = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if inspect.returncode and pull:
        downloaded = subprocess.run(
            ["docker", "pull", image], capture_output=True, text=True, timeout=300
        )
        if downloaded.returncode:
            raise RuntimeError("Docker image pull failed; check engine connectivity and image name")
        inspect = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    if inspect.returncode:
        raise RuntimeError(
            f"image {image} unavailable or Docker engine stopped; start Docker and enable image pull once"
        )
    output = Path(output_dir).resolve()
    snapshot = output / "workspace"
    count = copy_project(project, snapshot)
    name = "llm-optimise-" + uuid.uuid4().hex[:12]
    argv = container_command(image, snapshot, name, command, memory_mib, cpus, network)
    cancel = cancel or threading.Event()
    started, timed_out, cancelled = time.monotonic(), False, False
    output_limit_exceeded = False
    stdout_path, stderr_path = output / "stdout.log", output / "stderr.log"
    with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL)
        try:
            while process.poll() is None:
                timed_out = time.monotonic() - started > timeout_s
                cancelled = cancel.is_set()
                output_limit_exceeded = (
                    stdout_path.stat().st_size + stderr_path.stat().st_size > 16 * 1024**2
                )
                if timed_out or cancelled or output_limit_exceeded:
                    break
                time.sleep(0.1)
        finally:
            # Remove only our uniquely named container, even when the client times out or detaches.
            try:
                subprocess.run(
                    ["docker", "rm", "--force", name], capture_output=True, timeout=15, check=False
                )
            except (subprocess.TimeoutExpired, OSError):
                pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    return {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "cancelled": cancelled,
        "output_limit_exceeded": output_limit_exceeded,
        "stdout": stdout_path.read_bytes()[-128 * 1024 :].decode("utf-8", errors="replace"),
        "stderr": stderr_path.read_bytes()[-128 * 1024 :].decode("utf-8", errors="replace"),
        "image": image,
        "image_id": inspect.stdout.strip(),
        "command": command,
        "limits": {
            "memory_mib": memory_mib,
            "cpus": cpus,
            "timeout_s": timeout_s,
            "network": network,
            "pids": 128,
        },
        "elapsed_s": time.monotonic() - started,
        "copied_files": count,
        "artifact_dir": str(snapshot),
        "evidence": "Docker process execution; inspect logs for test count and assertions",
    }
