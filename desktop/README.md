# Native desktop application

This is a real Tauri v2 shell with a native window and tray. It launches the user's selected Python environment as an owned sidecar, obtains a private loopback URL from that process, and displays the existing LLM-Optimise interface in a native webview. The Python application, model runtimes and cloud credentials remain configured by the user.

## Build and run

Install the Python application in a dedicated environment first:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
npm ci --prefix desktop
npm run build --prefix desktop -- --bundles app
```

On macOS, the built application is `desktop/src-tauri/target/release/bundle/macos/LLM-Optimise.app`. Select the absolute Python executable path and an experiment workspace in its launcher. The chosen Python environment must contain `llm-optimise`; the desktop package does not silently download Python, models or API dependencies.

`python scripts/native-package-desktop.py` creates `desktop/profile-local/LLM-Optimise-macos-arm64.dmg`, verifies its checksum, mounts only that image read-only to inspect the app/bridge/Applications shortcut, then detaches it. This avoids Finder automation; Tauri's optional Finder-decoration script failed on the tested host, while this standard DMG path passed.

For development, use `npm run dev --prefix desktop`. `npm run icon --prefix desktop` converts the existing application artwork into native packaging formats. The generated macOS `.icns`, Windows `.ico` and PNG icons are included.

Build on the target platform:

| Target | Prerequisites and package route |
|---|---|
| macOS | Rust 1.95+, Node 20+, Xcode command-line tools. Build `app`; a DMG additionally uses `hdiutil` and Tauri's packaging script. |
| Linux | Rust 1.95+, Node 20+, compiler/linker, WebKitGTK 4.1, GTK3, librsvg and AppIndicator development packages as listed in Tauri's prerequisites. Build on the intended Linux distribution; AppImage/deb/rpm compatibility requires testing there. |
| Windows | Rust MSVC toolchain 1.95+, Node 20+, Visual Studio C++ Build Tools/Windows SDK, WebView2 runtime. Build MSI/NSIS on Windows and test installation there. |

See [Tauri platform prerequisites](https://v2.tauri.app/start/prerequisites/) and [distribution guidance](https://v2.tauri.app/distribute/). A Mac build does not validate Windows/Linux installers. Developer bundles are not a signed/notarized public release; signing identities, notarization and release CI must be configured for distribution.

## Lifecycle and security

- Closing the laboratory window hides it; the tray can show it again. **Quit and stop owned models** shuts down the owned Python service and its model processes.
- **Stop local service** closes the laboratory window and asks Python to cancel jobs/unload owned models. A bounded process-group/job cleanup follows if shutdown does not complete.
- The Rust process owner is shared with the optional supervisor. The service never binds to a public interface.
- The launcher is the only window with native IPC capabilities. The laboratory's loopback webview has no native IPC grants and navigation is restricted to that service origin.
- The interpreter is selected explicitly by absolute path and launched with Python's isolated mode. No shell command interpolation is used.
- Interpreter/workspace preferences are saved in the application's configuration directory. Service stderr is retained in `<workspace>/.llm-optimise/desktop/service.log`.

Selecting a Python executable is permission to execute that environment; use an environment you trust. Generated project code continues to run through the application's resource-bounded test workflow.

## Validation

`python desktop/validate.py` launches the built macOS app with a dedicated local workspace, waits for the real Tauri webview to load, checks the Python service's HTML and state API, exits the app, and verifies that the owned listening port is closed. It records results under `desktop/profile-local/`. It requires a logged-in macOS GUI session and a Python environment containing the application.

The built macOS arm64 app passed those live checks. Evidence: [native window and service validation](validation/macos-arm64-desktop.json), [verified disk-image layout and checksum](validation/macos-arm64-package.json). Windows/Linux desktop bundles have not been built or launched in this validation; native supervisor validation on Windows is a separate result.

The test uses these optional environment overrides, which also help automated development:

```text
LLM_OPTIMISE_PYTHON                  absolute interpreter path
LLM_OPTIMISE_WORKSPACE               absolute workspace path
LLM_OPTIMISE_DESKTOP_SETTINGS        preference file override
LLM_OPTIMISE_DESKTOP_AUTOSTART       start with the above selections
LLM_OPTIMISE_DESKTOP_TEST_REPORT     write native page-load evidence
LLM_OPTIMISE_DESKTOP_SMOKE_EXIT      exit after that page-load event
```

The native window's memory alone is not the complete app footprint: Python, models and WebKit helper processes must be accounted for separately. No total-memory or inference-speed advantage is claimed from desktop packaging.
