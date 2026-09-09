"""Build a relocatable, checksum-locked application runtime for the current host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "desktop/src-tauri/resources/runtime"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def host_target():
    system = {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}.get(platform.system())
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "AMD64": "x64"}.get(
        platform.machine()
    )
    if not system or not arch:
        raise ValueError("No bundled runtime is configured for this host")
    return f"{system}-{arch}"


def fetch_archive(source, cache):
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (source["sha256"] + ".tar.gz")
    if target.exists() and digest(target) == source["sha256"]:
        return target
    if not source["url"].startswith(
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
    ):
        raise ValueError("Runtime source must be the pinned official HTTPS release")
    partial = target.with_suffix(".partial")
    request = Request(source["url"], headers={"User-Agent": "LLM-Optimise-installer-builder"})
    try:
        with urlopen(request, timeout=90) as response, partial.open("wb") as stream:
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > source["bytes"]:
                    raise ValueError("Runtime archive exceeds pinned size")
                stream.write(chunk)
        if partial.stat().st_size != source["bytes"] or digest(partial) != source["sha256"]:
            raise ValueError("Runtime archive checksum or size mismatch")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def extract_archive(archive, target):
    # Python 3.12's data filter rejects device nodes, absolute paths and escaping links.
    with tarfile.open(archive, "r:gz") as bundle:
        if any(
            not m.name.startswith("python/") and m.name != "python" for m in bundle.getmembers()
        ):
            raise ValueError("Unexpected archive root")
        bundle.extractall(target, filter="data")


def write_launchers(output, windows):
    for name, module in (
        ("llm-optimise", "llm_optimise.cli"),
        ("llm-optimise-mcp", "llm_optimise.mcp_server"),
    ):
        if windows:
            content = f'@echo off\r\n"%~dp0python\\python.exe" -I -m {module} %*\r\nexit /b %errorlevel%\r\n'
            (output / (name + ".cmd")).write_bytes(content.encode())
        else:
            content = f'#!/bin/sh\nset -eu\nruntime_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexec "$runtime_dir/python/bin/python3" -I -m {module} "$@"\n'
            path = output / name
            path.write_text(content)
            path.chmod(0o755)


def clean_environment(cache):
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "UV_PYTHON"):
        env.pop(key, None)
    env.update(UV_CACHE_DIR=str(cache), UV_PYTHON_DOWNLOADS="never", PYTHONDONTWRITEBYTECODE="1")
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=ROOT / ".work/installer-cache")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 12):
        parser.error(
            "The installer builder requires Python 3.12+; installed users need no Python setup"
        )
    output, cache = args.output.resolve(), args.cache.resolve()
    uv = shutil.which("uv")
    if not uv:
        parser.error("Install uv in the build environment first")
    pinned = json.loads((ROOT / "packaging/python-runtime.json").read_text())
    target = host_target()
    source = pinned["platforms"][target]
    archive = fetch_archive(source, cache / "downloads")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build in a private sibling, validate relocation, then replace only our marked output.
    if output.exists():
        marker = output / "manifest.json"
        if (
            not marker.is_file()
            or json.loads(marker.read_text()).get("kind") != "llm-optimise-bundled-runtime"
        ):
            raise ValueError(
                "Refusing to replace a directory not marked as an LLM-Optimise runtime"
            )
    with tempfile.TemporaryDirectory(prefix="runtime-stage-", dir=output.parent) as temp:
        stage = Path(temp)
        extract_archive(archive, stage)
        python_relative = (
            "python/python.exe" if target.startswith("windows") else "python/bin/python3"
        )
        python = stage / python_relative
        if not python.is_file():
            raise ValueError("Pinned archive does not contain the expected interpreter")
        env = clean_environment(cache / "uv")

        def run(cmd):
            return subprocess.run([str(x) for x in cmd], env=env, cwd=ROOT, check=True)

        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                python,
                "--require-hashes",
                "--only-binary",
                ":all:",
                "-r",
                ROOT / "packaging/runtime-requirements.txt",
            ]
        )
        wheel_dir = stage / "build-wheel"
        run([uv, "build", "--wheel", "--out-dir", wheel_dir, ROOT])
        wheel = next(wheel_dir.glob("llm_optimise-*.whl"))
        run([uv, "pip", "install", "--python", python, "--no-deps", wheel])
        wheel_hash = digest(wheel)
        shutil.rmtree(wheel_dir)
        # Runtime-only packaging omits development tests, headers and static link archives.
        for relative in (
            "python/include",
            "python/Lib/test",
            "python/lib/python3.12/test",
            "python/lib/python3.12/idlelib",
            "python/lib/python3.12/turtledemo",
        ):
            path = stage / relative
            if path.is_dir():
                shutil.rmtree(path)
        for path in (stage / "python").rglob("*.a"):
            path.unlink()
        for path in sorted((stage / "python").rglob("__pycache__"), reverse=True):
            shutil.rmtree(path, ignore_errors=True)
        write_launchers(stage, target.startswith("windows"))
        shutil.copytree(ROOT / "packaging/python-licenses", stage / "licenses/python-distribution")
        (stage / "LICENSE-LLM-Optimise.txt").write_text((ROOT / "LICENSE").read_text())
        (stage / "THIRD-PARTY-NOTICES.txt").write_text(
            "This runtime includes CPython from Astral python-build-standalone and the Python dependencies listed in manifest.json.\n"
            "Their licenses remain in python/LICENSE*, python/share/, and package *.dist-info directories.\n"
            "Source and licensing metadata: https://github.com/astral-sh/python-build-standalone\n"
            "The LLM-Optimise application source is MIT licensed. Models and accelerator runtimes are not bundled.\n"
        )
        inventory = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import importlib.metadata as m,json; print(json.dumps(sorted([{'name':d.metadata['Name'],'version':d.version} for d in m.distributions()],key=lambda d:d['name'].lower())))",
                ],
                env=env,
            )
        )
        manifest = {
            "kind": "llm-optimise-bundled-runtime",
            "format_version": 1,
            "interpreter": python_relative,
            "target": target,
            "python_version": pinned["python_version"],
            "source": source,
            "dependency_lock_sha256": digest(ROOT / "packaging/runtime-requirements.txt"),
            "application_wheel_sha256": wheel_hash,
            "packages": inventory,
            "models_bundled": False,
            "mcp_included": True,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Moving the complete tree exposes absolute build-time path dependencies.
        relocated = stage / "relocated runtime with spaces"
        payload = stage / "payload"
        payload.mkdir()
        for path in list(stage.iterdir()):
            if path != payload:
                path.rename(payload / path.name)
        payload.rename(relocated)
        python = relocated / python_relative
        code = "import sys,json,psutil,ssl,sqlite3,llm_optimise,mcp,jwt; from importlib.resources import files; assert (files('llm_optimise')/'static'/'app-icon.png').is_file(); assert (files('llm_optimise')/'data'/'tasks.jsonl').is_file(); print(json.dumps({'python':sys.version.split()[0],'app':llm_optimise.__version__,'prefix':sys.prefix}))"
        observed = json.loads(subprocess.check_output([str(python), "-I", "-c", code], env=env))
        assert observed["python"] == pinned["python_version"]
        assert Path(observed["prefix"]).is_relative_to(relocated)
        run([python, "-I", "-m", "llm_optimise.cli", "--help"])
        run([python, "-I", "-m", "llm_optimise.mcp_server", "--help"])
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(relocated), output)
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    print(
        json.dumps(
            {
                "output": str(output),
                "target": target,
                "runtime_bytes": size,
                "relocation_check": "passed",
                "application": observed["app"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
