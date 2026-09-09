"""Build a relocatable, checksum-locked application runtime for the current host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
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
        (
            "https://github.com/astral-sh/python-build-standalone/releases/download/",
            "https://github.com/openssl/openssl/releases/download/",
            "https://files.pythonhosted.org/packages/",
        )
    ):
        raise ValueError("Build source must be a pinned official HTTPS release")
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


def extract_archive(archive, target, root="python"):
    # Python 3.12's data filter rejects device nodes, absolute paths and escaping links.
    with tarfile.open(archive, "r:gz") as bundle:
        if any(not m.name.startswith(root + "/") and m.name != root for m in bundle.getmembers()):
            raise ValueError("Unexpected archive root")
        bundle.extractall(target, filter="data")


def write_launchers(output, windows):
    for name, module in (
        ("llm-optimise", "llm_optimise"),
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
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "UV_PYTHON",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
    ):
        env.pop(key, None)
    env.update(UV_CACHE_DIR=str(cache), UV_PYTHON_DOWNLOADS="never", PYTHONDONTWRITEBYTECODE="1")
    return env


def replace_cryptography_requirement(requirements, wheel, version):
    pattern = r"(?m)^cryptography==[^\n]*(?:\n[ \t]+[^\n]*)*"
    matches = re.findall(pattern, requirements)
    if len(matches) != 1 or not matches[0].startswith(f"cryptography=={version} "):
        raise ValueError("Source-build version differs from the runtime dependency lock")
    replacement = f"cryptography @ {wheel.resolve().as_uri()} \\\n    --hash=sha256:{digest(wheel)}"
    return re.sub(pattern, lambda _: replacement, requirements, count=1)


def macos_linkage(libraries, load_commands, architectures, expected_arch, minimum_os):
    if architectures.strip().split() != [expected_arch]:
        raise ValueError("Cryptography extension architecture differs from the target")
    dependencies = [
        line.strip().split(" (", 1)[0] for line in libraries.splitlines()[1:] if line.strip()
    ]
    # `otool -L` includes LC_ID_DYLIB itself; that install name is not a load dependency.
    identities = [
        match.group(1)
        for block in re.findall(
            r"\bcmd LC_ID_DYLIB\n(.*?)(?=\nLoad command|\Z)", load_commands, re.S
        )
        if (match := re.search(r"(?m)^\s+name (.+?) \(offset", block))
    ]
    dependencies = [value for value in dependencies if value not in identities]
    if not dependencies or any(
        not value.startswith(("/usr/lib/", "/System/Library/"))
        or Path(value).name.startswith(("libssl", "libcrypto"))
        for value in dependencies
    ):
        raise ValueError("Cryptography must link only system libraries; OpenSSL must be static")
    blocks = re.findall(
        r"\bcmd LC_(?:BUILD_VERSION|VERSION_MIN_MACOSX)\n(.*?)(?=\nLoad command|\Z)",
        load_commands,
        re.S,
    )
    versions = [
        match.group(1)
        for block in blocks
        if (match := re.search(r"(?m)^\s+(?:minos|version)\s+(\d+(?:\.\d+)+)", block))
    ]
    limit = tuple(int(part) for part in minimum_os.split("."))
    if not versions or any(
        tuple(int(p) for p in version.split("."))[:2] > limit for version in versions
    ):
        raise ValueError("Cryptography extension exceeds the declared macOS deployment target")
    return {
        "libraries": dependencies,
        "install_names": identities,
        "architecture": expected_arch,
        "minimum_os": versions,
    }


def inspect_macos_cryptography(python, env, target, minimum_os):
    extension = Path(
        subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import cryptography.hazmat.bindings._rust as r; print(r.__file__)",
            ],
            env=env,
            text=True,
            timeout=30,
        ).strip()
    )

    def inspect(*arguments):
        return subprocess.check_output([*arguments, str(extension)], env=env, text=True, timeout=30)

    return macos_linkage(
        inspect("otool", "-L"),
        inspect("otool", "-l"),
        inspect("lipo", "-archs"),
        "x86_64" if target == "macos-x64" else "arm64",
        minimum_os,
    )


def build_macos_cryptography(python, stage, cache, uv, target):
    """Build pinned crypto/OpenSSL in private build directories, never Homebrew."""
    if target not in {"macos-x64", "macos-arm64"}:
        raise ValueError("This source-build path is only for macOS")
    pinned = json.loads((ROOT / "packaging/cryptography-source.json").read_text())
    lock = ROOT / "packaging/cryptography-build-requirements.txt"
    build = stage / "build-cryptography"
    build.mkdir()
    for name in ("openssl", "cryptography"):
        source = pinned[name]
        extract_archive(fetch_archive(source, cache / "downloads"), build, source["root"])
    openssl = build / pinned["openssl"]["root"]
    crypto = build / pinned["cryptography"]["root"]
    prefix = build / "openssl-static"
    env = clean_environment(cache / "uv")
    for name in list(env):
        if name.startswith(("OPENSSL_", "DYLD_", "PYO3_")) or name in {
            "CFLAGS",
            "CPPFLAGS",
            "LDFLAGS",
            "RUSTFLAGS",
            "ARCHFLAGS",
            "LIBRARY_PATH",
            "CPATH",
            "PKG_CONFIG_PATH",
            "CARGO_BUILD_TARGET",
            "CRYPTOGRAPHY_SUPPRESS_LINK_FLAGS",
        }:
            env.pop(name, None)
    minimum = pinned["macos_deployment_target"]
    env.update(
        MACOSX_DEPLOYMENT_TARGET=minimum,
        CFLAGS=f"-mmacosx-version-min={minimum}",
        LDFLAGS=f"-mmacosx-version-min={minimum}",
        RUSTFLAGS=f"-C link-arg=-mmacosx-version-min={minimum}",
        OPENSSL_STATIC="1",
        OPENSSL_NO_VENDOR="1",
        OPENSSL_DIR=str(prefix),
        CARGO_HOME=str(cache / "cargo"),
        CARGO_TARGET_DIR=str(build / "cargo-target"),
        CARGO_BUILD_JOBS="2",
    )

    def run(arguments, cwd=ROOT):
        subprocess.run([str(x) for x in arguments], cwd=cwd, env=env, check=True, timeout=1200)

    configure = "darwin64-x86_64-cc" if target == "macos-x64" else "darwin64-arm64-cc"
    run(
        [
            "perl",
            "Configure",
            configure,
            "no-shared",
            "no-module",
            "no-tests",
            f"--prefix={prefix}",
            "--openssldir=/etc/ssl",
        ],
        openssl,
    )
    run(["make", "-j2"], openssl)
    run(["make", "install_sw"], openssl)
    if not (prefix / "lib/libcrypto.a").is_file() or list(prefix.rglob("*.dylib")):
        raise ValueError("OpenSSL source build did not produce static-only libraries")
    build_env = build / "environment"
    run([uv, "venv", "--python", python, build_env])
    build_python = build_env / "bin/python"
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            build_python,
            "--require-hashes",
            "--only-binary",
            ":all:",
            "-r",
            lock,
        ]
    )
    # The hash-pinned sdist includes Cargo.lock; maturin's locked=true preserves crate checksums.
    if not (crypto / "Cargo.lock").is_file():
        raise ValueError("Cryptography source must include Cargo.lock")
    wheels = build / "wheels"
    run(
        [
            uv,
            "build",
            "--wheel",
            "--no-build-isolation",
            "--python",
            build_python,
            "--out-dir",
            wheels,
            crypto,
        ]
    )
    found = list(wheels.glob(f"cryptography-{pinned['cryptography_version']}-*.whl"))
    if len(found) != 1:
        raise ValueError("Expected exactly one source-built cryptography wheel")
    notices = stage / "licenses/openssl"
    notices.mkdir(parents=True)
    shutil.copy2(openssl / "LICENSE.txt", notices / "LICENSE.txt")
    report = {
        "name": "cryptography",
        "version": pinned["cryptography_version"],
        "source": pinned["cryptography"],
        "openssl_source": pinned["openssl"],
        "openssl_static": True,
        "openssl_version": pinned["openssl_version"],
        "macos_deployment_target": minimum,
        "build_dependency_lock_sha256": digest(lock),
        "cargo_lock_sha256": digest(crypto / "Cargo.lock"),
        "wheel_sha256": digest(found[0]),
        "wheel": found[0].name,
    }
    return found[0], report


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

        requirements = ROOT / "packaging/runtime-requirements.txt"
        source_builds = []
        if target == "macos-x64":
            crypto_wheel, crypto_report = build_macos_cryptography(python, stage, cache, uv, target)
            requirements = stage / "resolved-runtime-requirements.txt"
            requirements.write_text(
                replace_cryptography_requirement(
                    (ROOT / "packaging/runtime-requirements.txt").read_text(),
                    crypto_wheel,
                    crypto_report["version"],
                )
            )
            source_builds.append(crypto_report)
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
                requirements,
            ]
        )
        if source_builds:
            requirements.unlink()
            shutil.rmtree(stage / "build-cryptography")
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
        linux_pruning = None
        if target.startswith("linux-"):
            pruning_report = stage / "linux-runtime-pruning.json"
            run(
                [
                    sys.executable,
                    ROOT / "scripts/prune-linux-runtime.py",
                    "--runtime",
                    stage,
                    "--output",
                    pruning_report,
                ]
            )
            linux_pruning = json.loads(pruning_report.read_text())
            if linux_pruning.get("status") != "passed":
                raise ValueError("Linux runtime pruning and linkage audit did not pass")
            pruning_report.unlink()
        write_launchers(stage, target.startswith("windows"))
        shutil.copytree(ROOT / "packaging/python-licenses", stage / "licenses/python-distribution")
        (stage / "LICENSE-LLM-Optimise.txt").write_text((ROOT / "LICENSE").read_text())
        (stage / "THIRD-PARTY-NOTICES.txt").write_text(
            "This runtime includes CPython from Astral python-build-standalone and the Python dependencies listed in manifest.json.\n"
            "Their licenses remain in python/LICENSE*, python/share/, and package *.dist-info directories.\n"
            "Source-built OpenSSL licensing, when present, is in licenses/openssl/LICENSE.txt.\n"
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
            "source_built_dependencies": source_builds,
            "models_bundled": False,
            "mcp_included": True,
        }
        if linux_pruning is not None:
            manifest["linux_runtime_pruning"] = linux_pruning
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
        if source_builds:
            source_builds[0]["relocated_linkage"] = inspect_macos_cryptography(
                python, env, target, source_builds[0]["macos_deployment_target"]
            )
            crypto_check = "import cryptography; from cryptography.hazmat.backends.openssl.backend import backend; from cryptography.hazmat.primitives.asymmetric import rsa,padding; from cryptography.hazmat.primitives import hashes; key=rsa.generate_private_key(public_exponent=65537,key_size=2048); signature=key.sign(b'installer relocation',padding.PKCS1v15(),hashes.SHA256()); key.public_key().verify(signature,b'installer relocation',padding.PKCS1v15(),hashes.SHA256()); print(cryptography.__version__); print(backend.openssl_version_text())"
            source_builds[0]["relocated_crypto_smoke"] = (
                subprocess.check_output(
                    [str(python), "-I", "-c", crypto_check],
                    env=env,
                    text=True,
                    timeout=30,
                )
                .strip()
                .splitlines()
            )
            smoke = source_builds[0]["relocated_crypto_smoke"]
            if smoke[0] != source_builds[0]["version"] or not smoke[1].startswith(
                "OpenSSL " + source_builds[0]["openssl_version"] + " "
            ):
                raise ValueError("Relocated cryptography/OpenSSL versions differ from source pins")
            (relocated / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for module in ("llm_optimise", "llm_optimise.mcp_server"):
            help_text = subprocess.check_output(
                [str(python), "-I", "-m", module, "--help"], env=env, text=True, timeout=30
            )
            expected = "workbench" if module == "llm_optimise" else "--workspace"
            if "usage:" not in help_text or expected not in help_text:
                raise ValueError(f"Bundled {module} did not expose its command-line interface")
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
