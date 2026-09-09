# Native desktop application

The Tauri v2 desktop app opens the existing LLM-Optimise interface in a native webview and owns its private local Python service. Standalone installers include a Python runtime with the application and MCP dependencies. Users can also select an existing trusted Python environment. Models, inference engines, training environments and credentials remain user configured.

For normal installation, use the [installation guide](../docs/install.md). This page describes building, packaging and validating the desktop shell.

## Build a standalone installer

Build on the target platform and architecture. Development prerequisites are Python 3.12, uv 0.10.4, Rust 1.95.0, Node.js 22 and the native platform tools below. The runtime builder downloads its separately pinned Python 3.12.14 distribution and installs the locked application dependencies; it does not package your development virtual environment.

| Platform | Native build requirements | Output |
|---|---|---|
| macOS | Xcode command-line tools, `hdiutil` | App bundle and custom `.dmg` |
| Windows x64 | MSVC Rust toolchain, Visual Studio C++ build tools and Windows SDK | NSIS installer for the current user |
| Linux | Compiler/linker, WebKitGTK 4.1, GTK, Ayatana AppIndicator, OpenSSL, librsvg, libxdo and packaging utilities | `.deb` and `.AppImage` |

Use the [current Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) for your distribution. The [installer workflow](../.github/workflows/installers.yml) records the exact CI packages and tool versions.

From the repository root:

```bash
npm ci --prefix desktop
python scripts/build-bundled-runtime.py
npm run build --prefix desktop -- --config src-tauri/tauri.bundle.conf.json -- --locked
```

The build creates `desktop/src-tauri/resources/runtime/` with the interpreter, installed application, manifest, launchers and runtime licenses. The release-only configuration maps that directory to `runtime/` in the installed app resources, alongside `service.py`. Platform configurations select `app` on macOS, `nsis` on Windows and `deb,appimage` on Linux. Normal `cargo check` does not require the generated runtime directory.

Package the matching target, for example:

```bash
python scripts/package-installers.py --platform macos --arch arm64 --version 0.1.1 --output dist/installers
python scripts/verify-installer.py --artifacts dist/installers --output dist/installers/validation-macos-arm64.json
```

Use `macos`, `windows` or `linux` and the host's `arm64` or `x64` architecture. Run Linux verification inside a graphical session, or use CI's isolated virtual display:

```bash
xvfb-run -a dbus-run-session -- python scripts/verify-installer.py --artifacts dist/installers --output dist/installers/validation-linux-x64.json
```

The macOS packager creates a standard disk image and inspects its layout without Finder automation. Windows validation installs NSIS into an isolated per-user directory, launches that installation, then uninstalls it. Linux validation installs the Debian package and separately extracts and launches the AppImage. These verification commands perform real installation/removal actions on their host; CI supplies disposable hosts.

The [release procedure](../release/README.md) explains artifact acceptance and publication. This repository has no configured Apple notarization identity, Windows code-signing certificate or automatic updater. A local/ad-hoc signature is not a verified publisher signature.

## Develop the source launcher

Create a dedicated Python environment containing the app, then run the desktop development shell:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[mcp]'
npm ci --prefix desktop
npm run dev --prefix desktop
```

On Windows, create the environment with `py -3.12 -m venv .venv` and use `.venv\Scripts\python.exe`. Choose the custom-runtime option in the launcher and select the environment's absolute interpreter path. Source builds without bundled resources require this selection.

`npm run build --prefix desktop -- -- --locked` builds a source launcher using the native platform packaging configuration. Add the explicit bundle configuration only after generating the runtime. `npm run icon --prefix desktop` regenerates native icons from the application artwork.

## Runtime ownership and security

- The service listens only on a private loopback address. The desktop owns that service and its model process groups.
- Closing the laboratory window hides it; the tray can show it again. **Quit and stop owned models** requests shutdown and bounded cleanup.
- **Stop local service** closes the laboratory window, cancels jobs and unloads owned models. The process owner performs bounded cleanup if cooperative shutdown does not finish.
- Only the launcher has native IPC capabilities. The laboratory webview has no native IPC grants, and its navigation is restricted to its local service origin.
- Python runs in isolated mode with argument arrays, without shell interpolation. Choosing a custom interpreter authorizes executing that environment.
- Runtime and workspace preferences live in the application's configuration directory. Service errors are retained under `<workspace>/.llm-optimise/desktop/service.log`.

Workspaces belong outside the installation. Upgrading the app can replace bundled files; experiment results, projects and model configuration remain in the separately selected workspace. MCP can bridge the desktop's active service and reuse its existing job locks, model ownership and cancellation state; see [MCP setup](../docs/mcp.md).

## Native installation validation

`desktop/validate_installed.py` checks a specific installed binary:

```bash
python desktop/validate_installed.py --binary /absolute/path/to/installed/binary --workspace /absolute/path/to/isolated/workspace --output /absolute/path/to/report.json --timeout 60
```

The validator requires an empty isolated workspace, starts with fresh preferences and clears custom-interpreter overrides. It requires the packaged interpreter identified by the manifest, observes a real Tauri page-load event, checks the local HTML/CSRF/state API, and verifies service cleanup after a stop-file handshake. The JSON report is written to the requested location; an adjacent `<report-stem>-artifacts-<id>/` directory holds diagnostic logs. Windows paths can be quoted in PowerShell; Linux requires an active display or `xvfb-run`.

The older `python desktop/validate.py` remains a macOS developer smoke test using the calling Python environment. Its historical [native-window evidence](validation/macos-arm64-desktop.json) and [disk-image evidence](validation/macos-arm64-package.json) describe that earlier package. They do not substitute for the new installer acceptance reports.

These environment variables support development and validation:

```text
LLM_OPTIMISE_PYTHON                  absolute custom interpreter path
LLM_OPTIMISE_RUNTIME_MODE            bundled or custom interpreter selection
LLM_OPTIMISE_WORKSPACE               absolute workspace path
LLM_OPTIMISE_DESKTOP_SETTINGS        isolated preference-file override
LLM_OPTIMISE_DESKTOP_AUTOSTART       start with the selected settings
LLM_OPTIMISE_DESKTOP_TEST_REPORT     native page-load evidence path
LLM_OPTIMISE_DESKTOP_TEST_STOP       controlled smoke-test shutdown handshake
LLM_OPTIMISE_DESKTOP_SMOKE_EXIT      exit after the page-load test event
```

Desktop-window memory does not include the entire application footprint. Python, model processes and webview helpers contribute separately; packaging alone establishes no inference-speed or total-memory improvement.
