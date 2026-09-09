import importlib.util
from pathlib import Path
from subprocess import CompletedProcess

import pytest

SPEC = importlib.util.spec_from_file_location(
    "linux_runtime_prune", Path(__file__).resolve().parents[1] / "scripts/prune-linux-runtime.py"
)
prune = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prune)


def fixture_file(root, name, content=b"\x7fELFfixture"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_prune_removes_only_toolkit_and_preserves_core(tmp_path):
    python = fixture_file(tmp_path, "python/bin/python3")
    removed = [
        fixture_file(tmp_path, name)
        for name in (
            "python/lib/python3.12/lib-dynload/_tkinter.cpython-312-aarch64-linux-gnu.so",
            "python/lib/libtcl9tk9.0.so",
            "python/lib/itcl4.3.8/libtcl9itcl4.3.8.so",
            "python/lib/thread3.0.6/libtcl9thread3.0.6.so",
            "python/lib/tcl9.0/init.tcl",
            "python/lib/python3.12/tkinter/__init__.py",
        )
    ]
    core = fixture_file(tmp_path, "python/lib/python3.12/lib-dynload/_ssl.so")
    notice = fixture_file(tmp_path, "licenses/python-distribution/TCL.txt", b"license")
    idle = fixture_file(tmp_path, "python/bin/idle3.12")
    (idle.parent / "idle3").symlink_to("idle3.12")
    removed.append(idle)
    result = prune.prune_toolkit(tmp_path)
    assert result["removed_files"] == len(removed)
    assert result["removed_bytes"] == len(removed) * len(b"\x7fELFfixture")
    assert all(not path.exists() for path in removed)
    assert python.exists() and core.exists() and notice.exists()
    assert prune.prune_toolkit(tmp_path)["removed_files"] == 0


def test_prune_rejects_escape_before_any_removal(tmp_path):
    runtime, outside = tmp_path / "runtime", tmp_path / "outside"
    fixture_file(runtime, "python/bin/python3")
    preserved = fixture_file(runtime, "python/lib/libtcl9.0.so")
    outside.mkdir()
    (runtime / "python/lib/tk9.0").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prune.prune_toolkit(runtime)
    assert preserved.exists()


def test_missing_dependency_fails_with_relative_path(tmp_path):
    fixture_file(tmp_path, "python/bin/python3")
    with pytest.raises(ValueError, match="python/bin/python3: libexample.so"):
        prune.audit_elf(
            tmp_path,
            lambda *args, **kwargs: CompletedProcess(args, 0, "libexample.so => not found\n", ""),
        )


def test_linkage_audit_deduplicates_owned_symlinks(tmp_path):
    binary = fixture_file(tmp_path, "python/bin/python3")
    (binary.parent / "python").symlink_to("python3")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0, "libc.so.6 => /lib/libc.so.6 (0x1000)\n", "")

    result = prune.audit_elf(tmp_path, run)
    assert result["status"] == "passed"
    assert result["dynamic_elf_files_checked"] == len(calls) == 1
