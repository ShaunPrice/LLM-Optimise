"""Run the built macOS app, check its real webview/service, then verify cleanup."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "profile-local"
OUTPUT.mkdir(exist_ok=True)
report = OUTPUT / "native-window.json"
report.unlink(missing_ok=True)
binary = (
    ROOT
    / "src-tauri/target/release/bundle/macos/LLM-Optimise.app/Contents/MacOS/llm-optimise-desktop"
)
env = dict(os.environ)
env.update(
    LLM_OPTIMISE_PYTHON=sys.executable,
    LLM_OPTIMISE_WORKSPACE=str(OUTPUT / "workspace"),
    LLM_OPTIMISE_DESKTOP_SETTINGS=str(OUTPUT / "settings.json"),
    LLM_OPTIMISE_DESKTOP_TEST_REPORT=str(report),
    LLM_OPTIMISE_DESKTOP_AUTOSTART="1",
    LLM_OPTIMISE_DESKTOP_SMOKE_EXIT="1",
)
started = time.perf_counter()
with (OUTPUT / "smoke.log").open("w") as log:
    child = subprocess.Popen([str(binary)], env=env, stdout=log, stderr=log)
    try:
        while not report.exists() and time.perf_counter() - started < 35:
            if child.poll() is not None:
                raise AssertionError("Desktop exited before its webview loaded; inspect smoke.log")
            time.sleep(0.01)
        observed = json.loads(report.read_text())
        url = observed["url"]
        with urlopen(url, timeout=2) as response:
            html = response.read().decode()
            assert response.status == 200 and "LLM Optimise" in html and "csrf-token" in html
            assert "__CSRF_TOKEN__" not in html
        with urlopen(url.rstrip("/") + "/api/state", timeout=2) as response:
            state = json.load(response)
            assert response.status == 200 and "hardware" in state
        loaded_seconds = time.perf_counter() - started
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
address = urlparse(url)
with socket.socket() as probe:
    probe.settimeout(1)
    assert probe.connect_ex((address.hostname, address.port)) != 0
result = {
    "platform": sys.platform,
    "native_tauri_webview_loaded": True,
    "python_sidecar_http_passed": True,
    "application_state_http_passed": True,
    "owned_service_closed_after_app_exit": True,
    "load_and_http_check_seconds": loaded_seconds,
    "python_environment": "selected application virtual environment",
    "binary": "macOS arm64 app bundle",
    "signed_for_distribution": False,
    "notarized": False,
}
(OUTPUT / "desktop-validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
