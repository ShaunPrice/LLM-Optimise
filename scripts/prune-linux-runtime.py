"""Remove unused Tcl/Tk from the private Linux runtime and audit ELF linkage."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path

# Anchored to the standalone distribution's toolkit directories, never system paths.
TOOLKIT_PATTERNS = (
    "python/lib/python3.*/tkinter",
    "python/lib/python3.*/lib-dynload/_tkinter*.so",
    "python/lib/python3.*/idlelib",
    "python/lib/python3.*/turtledemo",
    "python/lib/python3.*/turtle.py",
    "python/bin/idle3*",
    "python/lib/libtcl*.so*",
    "python/lib/libtk*.so*",
    "python/lib/tcl[0-9]*",
    "python/lib/tk[0-9]*",
    "python/lib/itcl[0-9]*",
    "python/lib/thread[0-9]*",
)


def prune_toolkit(runtime: Path) -> dict:
    runtime = runtime.resolve()
    interpreter = runtime / "python/bin/python3"
    if not interpreter.is_file() or not interpreter.resolve().is_relative_to(runtime):
        raise ValueError("Expected an isolated Linux runtime containing python/bin/python3")
    with interpreter.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("Runtime interpreter must be a Linux ELF executable")
    targets = sorted({path for pattern in TOOLKIT_PATTERNS for path in runtime.glob(pattern)})
    # Validate the entire removal set before changing anything. Do not follow directory links.
    for path in targets:
        if not path.resolve().is_relative_to(runtime):
            raise ValueError("Toolkit removal target symlink must not escape the runtime")
        if path.is_dir() and any(child.is_symlink() for child in path.rglob("*")):
            raise ValueError("Toolkit directory unexpectedly contains a symlink")
    files = [
        file
        for path in targets
        for file in ([path] if path.is_file() else path.rglob("*"))
        if file.is_file() and not file.is_symlink()
    ]
    removed_bytes = sum(file.stat().st_size for file in files)
    removed = [str(path.relative_to(runtime)) for path in targets]
    for path in targets:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    return {"removed_paths": removed, "removed_files": len(files), "removed_bytes": removed_bytes}


def audit_elf(runtime: Path, runner=subprocess.run) -> dict:
    runtime = runtime.resolve()
    checked, static, seen = [], [], set()
    for path in sorted((runtime / "python").rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(runtime):
            raise ValueError("Runtime file symlink escapes the isolated runtime")
        if resolved in seen:
            continue
        seen.add(resolved)
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        relative = str(path.relative_to(runtime))
        result = runner(["ldd", str(path)], capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        if "not found" in output:
            missing = [
                line.strip().split(" =>", 1)[0]
                for line in output.splitlines()
                if "not found" in line
            ]
            raise ValueError(f"Unresolved ELF dependencies in {relative}: {', '.join(missing)}")
        if result.returncode:
            if "not a dynamic executable" in output or "statically linked" in output:
                static.append(relative)
                continue
            raise ValueError(f"ldd failed for {relative} with exit code {result.returncode}")
        checked.append(relative)
    if not checked:
        raise ValueError("No dynamic ELF files were audited")
    return {
        "status": "passed",
        "dynamic_elf_files_checked": len(checked),
        "dynamic_elf_paths": checked,
        "static_elf_paths": static,
        "unresolved_dependencies": [],
        "scope": "ldd resolution of remaining ELF files on the build host; native installer launch is verified separately. Libraries opened only through dlopen are not covered.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run this tool on the Linux target architecture")
    report = {"schema": "llm-optimise-linux-runtime-prune/v1", "status": "failed"}
    try:
        report.update(prune_toolkit(args.runtime))
        report["linkage"] = audit_elf(args.runtime)
        report["status"] = "passed"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc).replace(str(args.runtime.resolve()), "<runtime>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("schema", "status")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
