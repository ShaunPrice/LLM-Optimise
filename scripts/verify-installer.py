"""Install/extract a native package into an isolated location and prove its bundled runtime works."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def one(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"Expected one artifact, found {len(paths)}")
    return paths[0]


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, timeout=180, **kwargs)


def run_nsis(binary, path, *, uninstall=False):
    # NSIS requires its directory switch to be the final, unquoted command tail.
    # shell=False passes this directly to CreateProcess, without shell expansion.
    switch = "_?=" if uninstall else "/D="
    command = subprocess.list2cmdline([str(binary), "/S"]) + f" {switch}{path}"
    return subprocess.run(command, check=True, timeout=180, shell=False)


def runtime_checks(resources, workspace):
    runtime = resources / "runtime"
    manifest = json.loads((runtime / "manifest.json").read_text())
    python = runtime / manifest["interpreter"]
    if not python.resolve().is_relative_to(runtime.resolve()):
        raise ValueError("Bundled interpreter escapes the installed runtime")
    env = dict(os.environ)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "LLM_OPTIMISE_PYTHON",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
    ):
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = "import json,sys,psutil,mcp,jwt,llm_optimise,ssl,sqlite3; print(json.dumps({'python':sys.executable,'prefix':sys.prefix,'version':llm_optimise.__version__}))"
    observation = json.loads(subprocess.check_output([str(python), "-I", "-c", code], env=env))
    if not Path(observation["prefix"]).resolve().is_relative_to(runtime.resolve()):
        raise ValueError("Installed Python prefix is outside the package")
    run([python, "-I", "-m", "llm_optimise.cli", "--help"], env=env, stdout=subprocess.DEVNULL)
    run(
        [python, "-I", "-m", "llm_optimise.mcp_server", "--help"],
        env=env,
        stdout=subprocess.DEVNULL,
    )
    return {
        "bundled_imports_passed": True,
        "cli_help_passed": True,
        "mcp_help_passed": True,
        "interpreter_relative": manifest["interpreter"],
        "python_version": manifest["python_version"],
        "app_version": observation["version"],
        "independent_of_system_python": True,
    }


def native_check(binary, resources, workspace, output):
    output.mkdir(parents=True, exist_ok=True)
    result = runtime_checks(resources, workspace)
    report = output / "native.json"
    run(
        [
            sys.executable,
            ROOT / "desktop/validate_installed.py",
            "--binary",
            binary,
            "--workspace",
            workspace,
            "--output",
            report,
            "--timeout",
            "60",
        ]
    )
    result["native"] = json.loads(report.read_text())
    (workspace / "preserve-my-results.txt").write_text(
        "User workspace must survive application uninstall.\n"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifacts, output = args.artifacts.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = output.with_name(output.stem + "-artifacts")
    evidence.mkdir(exist_ok=True)
    system = platform.system()
    result = {"platform": system, "architecture": platform.machine(), "passed": False, "checks": {}}
    try:
        with tempfile.TemporaryDirectory(prefix="LLM Optimise installer smoke ") as temp:
            root = Path(temp)
            workspace = root / "Research workspace"
            workspaces = []
            if system == "Darwin":
                package = one(artifacts.glob("*.dmg"))
                mount = root / "mounted"
                mount.mkdir()
                run(["hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", mount, package])
                try:
                    if not (mount / "Applications").is_symlink():
                        raise ValueError("DMG Applications shortcut missing")
                    app = root / "Installed Applications/LLM-Optimise.app"
                    shutil.copytree(mount / "LLM-Optimise.app", app, symlinks=True)
                finally:
                    run(["hdiutil", "detach", mount])
                result["checks"]["dmg"] = native_check(
                    app / "Contents/MacOS/llm-optimise-desktop",
                    app / "Contents/Resources",
                    workspace,
                    evidence,
                )
                workspaces.append(workspace)
                shutil.rmtree(app)
                result["installation"] = (
                    "Mounted verified DMG, copied app to a relocated path with spaces, launched and removed app"
                )
            elif system == "Windows":
                package = one(artifacts.glob("*-setup.exe"))
                install = root / "Installed Application"
                run_nsis(package, install)
                binary = install / "llm-optimise-desktop.exe"
                if not binary.is_file():
                    raise ValueError("NSIS did not install the application at the requested path")
                try:
                    result["checks"]["nsis"] = native_check(binary, install, workspace, evidence)
                    workspaces.append(workspace)
                finally:
                    uninstaller = one(install.glob("*ninstall*.exe"))
                    run_nsis(uninstaller, install, uninstall=True)
                if binary.exists():
                    raise ValueError("NSIS uninstall did not remove the application")
                result["installation"] = (
                    "Per-user NSIS silent install, real native launch, and uninstall"
                )
            elif system == "Linux":
                deb = one(artifacts.glob("*.deb"))
                metadata = subprocess.check_output(
                    ["dpkg-deb", "-f", str(deb), "Package"], text=True
                ).strip()
                existing = subprocess.run(
                    ["dpkg-query", "-W", "-f=${db:Status-Status}", metadata],
                    capture_output=True,
                    text=True,
                )
                if existing.returncode == 0 and existing.stdout == "installed":
                    raise ValueError(
                        "Refusing to replace an existing installation during the smoke test"
                    )
                run(["sudo", "apt-get", "install", "-y", str(deb)])
                try:
                    files = subprocess.check_output(
                        ["dpkg-query", "-L", metadata], text=True
                    ).splitlines()
                    binary = one(Path(f) for f in files if f.endswith("/bin/llm-optimise-desktop"))
                    manifest = one(Path(f) for f in files if f.endswith("/runtime/manifest.json"))
                    result["checks"]["deb"] = native_check(
                        binary, manifest.parent.parent, workspace / "deb", evidence / "deb"
                    )
                    workspaces.append(workspace / "deb")
                finally:
                    run(["sudo", "apt-get", "remove", "-y", metadata])
                package = one(artifacts.glob("*.AppImage"))
                package.chmod(package.stat().st_mode | 0o111)
                extracted = root / "AppImage extracted"
                extracted.mkdir()
                run([package, "--appimage-extract"], cwd=extracted, stdout=subprocess.DEVNULL)
                appdir = extracted / "squashfs-root"
                manifest = one(appdir.rglob("runtime/manifest.json"))
                result["checks"]["appimage"] = native_check(
                    appdir / "AppRun",
                    manifest.parent.parent,
                    workspace / "AppImage",
                    evidence / "AppImage",
                )
                workspaces.append(workspace / "AppImage")
                shutil.rmtree(extracted)
                result["installation"] = (
                    "Native deb install/launch/remove; portable AppImage extraction/launch"
                )
            else:
                raise ValueError("Unsupported installer validation host")
            for checked_workspace in workspaces:
                if (
                    not (checked_workspace / "preserve-my-results.txt")
                    .read_text()
                    .startswith("User workspace")
                ):
                    raise ValueError("Uninstall changed the user workspace")
            result["workspace_preserved_after_uninstall"] = True
        result["passed"] = True
        result["artifacts"] = [
            {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(artifacts.iterdir())
            if p.suffix in (".dmg", ".deb", ".AppImage", ".exe")
        ]
        result["limitations"] = [
            "No distribution signing/notarization claim",
            "Native hosted-runner acceptance is distinct from inference or accelerator validation",
        ]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
