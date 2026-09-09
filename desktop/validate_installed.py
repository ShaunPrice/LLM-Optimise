"""Validate an installed native app with its bundled Python and real webview.

Run package installation/extraction separately, then pass the resulting binary.
Linux callers supply a graphical session, for example through xvfb-run.
The validator itself uses standard-library Python; the launched app must use its
own runtime. Detailed native paths stay in the adjacent local artifacts folder.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener


def _absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("use an absolute path")
    return path


def _process_alive(pid):
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: the PID no longer exists.
                return False
            if error == 5:  # Access denied is not proof that a process exited.
                return True
            raise OSError(error, "Cannot determine whether the owned sidecar exited")
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError(ctypes.get_last_error(), "Cannot query owned sidecar")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _canonical(value):
    # Windows APIs may report an extended-length prefix on only one side.
    value = str(value)
    if os.name == "nt" and value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value).resolve()


def _child_environment(artifacts, workspace):
    env = dict(os.environ)
    for name in (
        "LLM_OPTIMISE_PYTHON",
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    ):
        env.pop(name, None)
    # Remove developer interpreter locations while preserving normal OS helpers.
    # Actual bundled-executable identity is checked independently after startup.
    excluded = {_canonical(Path(sys.executable).parent)}
    for name in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if os.environ.get(name):
            prefix = Path(os.environ[name])
            excluded.update(
                {_canonical(prefix), _canonical(prefix / "bin"), _canonical(prefix / "Scripts")}
            )
    parts = [
        part
        for part in env.get("PATH", os.defpath).split(os.pathsep)
        if part and _canonical(part) not in excluded
    ]
    env["PATH"] = os.pathsep.join(parts)
    env.update(
        LLM_OPTIMISE_RUNTIME_MODE="bundled",
        LLM_OPTIMISE_WORKSPACE=str(workspace),
        LLM_OPTIMISE_DESKTOP_SETTINGS=str(artifacts / "settings.json"),
        LLM_OPTIMISE_DESKTOP_TEST_REPORT=str(artifacts / "native-window.json"),
        LLM_OPTIMISE_DESKTOP_TEST_STOP=str(artifacts / "stop-request"),
        LLM_OPTIMISE_DESKTOP_AUTOSTART="1",
        LLM_OPTIMISE_DESKTOP_SMOKE_EXIT="1",
    )
    return env


def validate(binary: Path, workspace: Path, output: Path, timeout=60):
    binary, workspace, output = binary.resolve(), workspace.resolve(), output.resolve()
    if not binary.is_file():
        raise ValueError("Installed binary does not exist")
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("Native installation validation requires an empty, dedicated workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = output.parent / f"{output.stem}-artifacts-{uuid.uuid4().hex[:8]}"
    artifacts.mkdir()
    report = artifacts / "native-window.json"
    stop = artifacts / "stop-request"
    env = _child_environment(artifacts, workspace)
    opener = build_opener(ProxyHandler({}))
    result = {
        "schema_version": 1,
        "platform": sys.platform,
        "architecture": platform.machine(),
        "binary_name": binary.name,
        "validation_scope": "installed binary, bundled runtime, native webview, HTTP and owned service cleanup",
        "status": "failed",
    }
    started = time.perf_counter()
    child = None
    observed = None
    try:
        with (artifacts / "native.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen([str(binary)], env=env, cwd=artifacts, stdout=log, stderr=log)
            while not report.exists():
                if child.poll() is not None:
                    raise AssertionError(
                        "Native app exited before its webview loaded; inspect native.log"
                    )
                if time.perf_counter() - started > timeout:
                    raise TimeoutError("Native webview did not load before the validation deadline")
                time.sleep(0.05)
            observed = json.loads(report.read_text(encoding="utf-8"))
            assert observed["event"] == "native_webview_loaded"
            assert observed["runtime_mode"] == "bundled"
            assert observed["sidecar"]["ready"] is True
            resources = _canonical(observed["resource_dir"])
            runtime = (resources / "runtime").resolve()
            manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["format_version"] == 1
            expected = (runtime / manifest["interpreter"]).resolve()
            actual = _canonical(observed["sidecar"]["python_executable"])
            resolved = _canonical(observed["resolved_interpreter"])
            assert expected.is_relative_to(runtime) and expected.is_file()
            assert actual == expected == resolved, (
                "App must use the interpreter from its current bundle"
            )
            assert observed["sidecar"]["sample_data_available"] is True
            assert _canonical(observed["sidecar"]["workspace"]) == workspace
            address = urlparse(observed["url"])
            assert address.scheme == "http" and address.hostname == "127.0.0.1" and address.port
            assert (
                not address.username
                and not address.password
                and not address.query
                and not address.fragment
            )
            url = f"http://127.0.0.1:{address.port}"
            with opener.open(url, timeout=10) as response:
                html = response.read().decode("utf-8")
                assert response.status == 200 and "LLM Optimise" in html and "csrf-token" in html
                assert "__CSRF_TOKEN__" not in html
            with opener.open(url + "/api/state", timeout=10) as response:
                state = json.load(response)
                assert response.status == 200 and "hardware" in state
                assert state["tasks"], "First launch must include the bundled sample dataset"
                assert not state["jobs"], "Validation must not start model jobs"
            settings = json.loads((artifacts / "settings.json").read_text(encoding="utf-8"))
            assert settings["runtime_mode"] == "bundled" and settings["interpreter"] == ""
            assert _canonical(settings["workspace"]) == workspace
            loaded_seconds = time.perf_counter() - started
            stop.write_text("HTTP checks complete; quit normally.\n", encoding="utf-8")
            assert child.wait(timeout=20) == 0
        pid = observed["sidecar"]["pid"]
        deadline = time.monotonic() + 5
        while _process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _process_alive(pid), "Owned Python sidecar must exit with the native app"
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", address.port)) != 0, (
                "Owned HTTP listener must close"
            )
        result.update(
            status="passed",
            native_tauri_webview_loaded=True,
            bundled_python_verified=True,
            explicit_python_override_absent="LLM_OPTIMISE_PYTHON" not in env,
            python_environment_injection_removed=not any(
                name in env for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX")
            ),
            interpreter_location="current application resource_dir/runtime",
            python_version=observed["sidecar"]["python_version"],
            package_version=observed["sidecar"]["package_version"],
            runtime_selection_persisted_without_absolute_interpreter=True,
            first_run_sample_tasks_available=True,
            python_sidecar_http_passed=True,
            application_state_http_passed=True,
            owned_sidecar_pid_exited=True,
            owned_service_listener_closed=True,
            load_and_http_check_seconds=loaded_seconds,
            elapsed_seconds=time.perf_counter() - started,
        )
        return result
    except Exception as exc:
        result["error"] = str(exc) or type(exc).__name__
        raise
    finally:
        if child is not None and child.poll() is None:
            stop.write_text("Validation ending; quit normally.\n", encoding="utf-8")
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=_absolute)
    parser.add_argument("--workspace", required=True, type=_absolute)
    parser.add_argument("--output", required=True, type=_absolute)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout must be 1..300 seconds")
    result = validate(args.binary, args.workspace, args.output, args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
