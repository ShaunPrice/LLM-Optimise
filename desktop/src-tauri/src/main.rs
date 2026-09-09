#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use llm_supervisor::OwnedProcess;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Component, Path, PathBuf},
    process::{ChildStdin, Command, Stdio},
    sync::{Mutex, mpsc},
    thread,
    time::Duration,
};
use tauri::{
    Manager, State, WebviewUrl, WebviewWindowBuilder,
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
};

#[derive(Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
enum RuntimeMode {
    #[default]
    Bundled,
    Custom,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
struct Settings {
    format_version: u32,
    runtime_mode: RuntimeMode,
    interpreter: String,
    workspace: String,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            format_version: 2,
            runtime_mode: RuntimeMode::Bundled,
            interpreter: String::new(),
            workspace: String::new(),
        }
    }
}
impl Settings {
    fn normalise(mut self) -> Self {
        self.format_version = 2;
        if self.runtime_mode == RuntimeMode::Bundled {
            self.interpreter.clear();
        }
        self
    }
}

#[derive(Serialize)]
struct LauncherDefaults {
    #[serde(flatten)]
    settings: Settings,
    bundled_available: bool,
    bundled_error: Option<String>,
}

#[derive(Deserialize)]
struct RuntimeManifest {
    format_version: u32,
    interpreter: String,
}

fn legacy_bundle_path(value: &str) -> bool {
    let value = value.replace('\\', "/").to_ascii_lowercase();
    [
        "/runtime/python/bin/python3",
        "/runtime/python/bin/python",
        "/runtime/python/python.exe",
    ]
    .iter()
    .any(|suffix| value.ends_with(suffix))
}

fn decode_settings(bytes: &[u8]) -> Result<Settings, String> {
    let document: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let mut settings: Settings =
        serde_json::from_value(document.clone()).map_err(|e| e.to_string())?;
    if document.get("runtime_mode").is_none() && !settings.interpreter.is_empty() {
        settings.runtime_mode = if legacy_bundle_path(&settings.interpreter) {
            RuntimeMode::Bundled
        } else {
            RuntimeMode::Custom
        };
    }
    Ok(settings.normalise())
}

fn bundled_python(resource_dir: &Path) -> Result<PathBuf, String> {
    let runtime = resource_dir.join("runtime");
    let manifest_path = runtime.join("manifest.json");
    let info = fs::metadata(&manifest_path).map_err(|_| {
        "This build has no bundled runtime. Install the full desktop package, or select a custom Python in Advanced settings.".to_string()
    })?;
    if info.len() > 65536 {
        return Err("The bundled runtime manifest is invalid. Reinstall the application.".into());
    }
    let manifest: RuntimeManifest =
        serde_json::from_slice(&fs::read(manifest_path).map_err(|e| e.to_string())?)
            .map_err(|_| "The bundled runtime manifest is invalid. Reinstall the application.")?;
    let relative = Path::new(&manifest.interpreter);
    if manifest.format_version != 1
        || manifest.interpreter.is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err("The bundled runtime manifest uses an unsupported format or path.".into());
    }
    let root = runtime.canonicalize().map_err(|e| e.to_string())?;
    let interpreter = root.join(relative).canonicalize().map_err(|_| {
        "The bundled Python is missing. Reinstall the application; no system Python is required.".to_string()
    })?;
    if !interpreter.is_file() || !interpreter.starts_with(&root) {
        return Err(
            "The bundled interpreter must be a file inside the application runtime.".into(),
        );
    }
    Ok(interpreter)
}

fn resolve_python(resource_dir: &Path, settings: &Settings) -> Result<PathBuf, String> {
    match settings.runtime_mode {
        RuntimeMode::Bundled => bundled_python(resource_dir),
        RuntimeMode::Custom => {
            let path = PathBuf::from(&settings.interpreter);
            if !path.is_absolute() || !path.is_file() {
                return Err(
                    "Select an existing custom Python executable using its absolute path.".into(),
                );
            }
            // Preserve venv executable identity rather than canonicalising its symlink.
            Ok(path)
        }
    }
}

struct Service {
    process: OwnedProcess,
    input: Option<ChildStdin>,
    url: String,
    settings: Settings,
    evidence: serde_json::Value,
}
impl Drop for Service {
    fn drop(&mut self) {
        if let Some(input) = self.input.as_mut() {
            let _ = writeln!(input, "stop");
            let _ = input.flush();
        }
        // Give the Python service an opportunity to cancel jobs and unload models.
        for _ in 0..120 {
            if self.process.exited().unwrap_or(true) {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        let _ = self.process.finish(Duration::from_millis(500));
    }
}
#[derive(Default)]
struct AppState {
    service: Mutex<Option<Service>>,
}

fn settings_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("LLM_OPTIMISE_DESKTOP_SETTINGS") {
        return Ok(PathBuf::from(path));
    }
    app.path()
        .app_config_dir()
        .map(|p| p.join("launcher.json"))
        .map_err(|e| e.to_string())
}
#[tauri::command]
fn defaults(app: tauri::AppHandle) -> LauncherDefaults {
    let mut value = settings_path(&app)
        .ok()
        .and_then(|p| fs::read(p).ok())
        .and_then(|b| decode_settings(&b).ok())
        .unwrap_or_default();
    if let Ok(python) = std::env::var("LLM_OPTIMISE_PYTHON")
        && !python.is_empty()
    {
        value.interpreter = python;
        value.runtime_mode = RuntimeMode::Custom;
    }
    if let Ok(mode) = std::env::var("LLM_OPTIMISE_RUNTIME_MODE") {
        match mode.as_str() {
            "bundled" => value.runtime_mode = RuntimeMode::Bundled,
            "custom" => value.runtime_mode = RuntimeMode::Custom,
            _ => (),
        }
    }
    if let Ok(workspace) = std::env::var("LLM_OPTIMISE_WORKSPACE") {
        value.workspace = workspace;
    }
    if value.workspace.is_empty() {
        value.workspace = app
            .path()
            .document_dir()
            .or_else(|_| app.path().app_data_dir())
            .unwrap_or_else(|_| std::env::temp_dir())
            .join("LLM-Optimise")
            .to_string_lossy()
            .into();
    }
    let runtime = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())
        .and_then(|resource| bundled_python(&resource));
    LauncherDefaults {
        settings: value.normalise(),
        bundled_available: runtime.is_ok(),
        bundled_error: runtime.err(),
    }
}

fn launch(app: &tauri::AppHandle, settings: &Settings) -> Result<Service, String> {
    let resources = app.path().resource_dir().map_err(|e| e.to_string())?;
    let python = resolve_python(&resources, settings)?;
    let workspace = Path::new(&settings.workspace);
    if !workspace.is_absolute() {
        return Err("The workspace must be an absolute directory path.".into());
    }
    fs::create_dir_all(workspace).map_err(|e| format!("Cannot create workspace: {e}"))?;
    let mut script = resources.join("service.py");
    if cfg!(debug_assertions) && !script.is_file() {
        script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/service.py");
    }
    if !script.is_file() {
        return Err(
            "The desktop service is missing. Reinstall the full application package.".into(),
        );
    }
    let logs = workspace.join(".llm-optimise").join("desktop");
    fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
    let log = fs::File::create(logs.join("service.log")).map_err(|e| e.to_string())?;
    let mut command = Command::new(&python);
    command
        .args(["-I", "-u"])
        .arg(script)
        .arg("--workspace")
        .arg(workspace)
        .current_dir(workspace)
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(log);
    let mut process =
        OwnedProcess::spawn(&mut command).map_err(|e| format!("Python could not start: {e}"))?;
    let output = process
        .child
        .stdout
        .take()
        .ok_or("Missing service output pipe")?;
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(output.take(65536))
            .read_line(&mut line)
            .map(|_| line);
        let _ = tx.send(result);
    });
    let line=rx.recv_timeout(Duration::from_secs(30)).map_err(|_|"Python startup timed out. Inspect .llm-optimise/desktop/service.log in the workspace.")?.map_err(|e|e.to_string())?;
    let result: serde_json::Value = serde_json::from_str(&line)
        .map_err(|_| "Python did not return a valid startup handshake. Inspect the service log.")?;
    if let Some(error) = result.get("error").and_then(|v| v.as_str()) {
        return Err(error.into());
    }
    let url = result
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or("Python did not report a service URL")?
        .to_string();
    let parsed: tauri::Url = url.parse().map_err(|_| "Invalid service URL")?;
    if parsed.scheme() != "http"
        || parsed.host_str() != Some("127.0.0.1")
        || parsed.port().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || parsed.path() != "/"
    {
        return Err("Service must use its own loopback-only HTTP endpoint.".into());
    }
    let input = process.child.stdin.take();
    Ok(Service {
        process,
        input,
        url,
        settings: settings.clone(),
        evidence: serde_json::json!({
            "runtime_mode": settings.runtime_mode,
            "resolved_interpreter": python,
            "resource_dir": resources,
            "sidecar": result,
        }),
    })
}

fn show_lab(app: &tauri::AppHandle, service: &Service) -> Result<(), String> {
    let url = &service.url;
    if let Some(window) = app.get_webview_window("lab") {
        // Recreate the window when a crashed service receives a new loopback port;
        // its navigation policy is bound to the previous service origin.
        if window
            .url()
            .map_err(|e| e.to_string())?
            .as_str()
            .trim_end_matches('/')
            == url.trim_end_matches('/')
        {
            window.show().map_err(|e| e.to_string())?;
            window.set_focus().map_err(|e| e.to_string())?;
            return Ok(());
        }
        window.destroy().map_err(|e| e.to_string())?;
    }
    let origin = url.to_string();
    let handle = app.clone();
    let evidence = service.evidence.clone();
    let window = WebviewWindowBuilder::new(
        app,
        "lab",
        WebviewUrl::External(url.parse().map_err(|_| "Invalid URL")?),
    )
    .title("LLM-Optimise · Laboratory")
    .inner_size(1320.0, 880.0)
    .min_inner_size(900.0, 620.0)
    .on_navigation(move |target| {
        target.as_str() == origin || target.as_str().starts_with(&(origin.clone() + "/"))
    })
    .on_page_load(move |_window, payload| {
        if payload.event() == tauri::webview::PageLoadEvent::Finished
            && let Some(path) = std::env::var_os("LLM_OPTIMISE_DESKTOP_TEST_REPORT")
        {
            let mut report = evidence.clone();
            report["event"] = "native_webview_loaded".into();
            report["url"] = payload.url().as_str().into();
            report["tauri"] = 2.into();
            let path = PathBuf::from(path);
            let temporary = path.with_extension("tmp");
            if let Ok(bytes) = serde_json::to_vec_pretty(&report)
                && fs::write(&temporary, bytes).is_ok()
            {
                let _ = fs::rename(temporary, path);
            }
            if std::env::var_os("LLM_OPTIMISE_DESKTOP_SMOKE_EXIT").is_some() {
                let app = handle.clone();
                thread::spawn(move || {
                    if let Some(stop) = std::env::var_os("LLM_OPTIMISE_DESKTOP_TEST_STOP") {
                        for _ in 0..1200 {
                            if Path::new(&stop).exists() {
                                break;
                            }
                            thread::sleep(Duration::from_millis(50));
                        }
                    } else {
                        thread::sleep(Duration::from_secs(5));
                    }
                    app.exit(0);
                });
            }
        }
    })
    .build()
    .map_err(|e| e.to_string())?;
    let copy = window.clone();
    window.on_window_event(move |event| {
        if let tauri::WindowEvent::CloseRequested { api, .. } = event {
            api.prevent_close();
            let _ = copy.hide();
        }
    });
    Ok(())
}

#[tauri::command]
async fn start_service(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    interpreter: Option<String>,
    workspace: String,
    runtime_mode: Option<RuntimeMode>,
) -> Result<serde_json::Value, String> {
    let selected = Settings {
        runtime_mode: runtime_mode.unwrap_or_default(),
        interpreter: interpreter.unwrap_or_default(),
        workspace,
        ..Settings::default()
    }
    .normalise();
    let mut service = state
        .service
        .lock()
        .map_err(|_| "Service state unavailable")?;
    if let Some(running) = service.as_mut() {
        if !running.process.exited().unwrap_or(true) {
            if running.settings.workspace != selected.workspace
                || running.settings.runtime_mode != selected.runtime_mode
                || running.settings.interpreter != selected.interpreter
            {
                return Err("The laboratory is already open with different settings. Stop its local service in Advanced before switching workspace or runtime.".into());
            }
            show_lab(&app, running)?;
            return Ok(serde_json::json!({"url":running.url}));
        }
        *service = None;
    }
    let running = launch(&app, &selected)?;
    let url = running.url.clone();
    let path = settings_path(&app)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(&selected).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    show_lab(&app, &running)?;
    *service = Some(running);
    Ok(serde_json::json!({"url":url}))
}
#[tauri::command]
fn stop_service(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("lab") {
        let _ = window.destroy();
    }
    *state
        .service
        .lock()
        .map_err(|_| "Service state unavailable")? = None;
    Ok(())
}

fn main() {
    let app = tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            defaults,
            start_service,
            stop_service
        ])
        .setup(|app| {
            let show = MenuItem::with_id(app, "show", "Show laboratory", true, None::<&str>)?;
            let settings =
                MenuItem::with_id(app, "settings", "Launcher settings", true, None::<&str>)?;
            let quit = MenuItem::with_id(
                app,
                "quit",
                "Quit and stop owned models",
                true,
                None::<&str>,
            )?;
            let menu = Menu::with_items(app, &[&show, &settings, &quit])?;
            let icon = app
                .default_window_icon()
                .cloned()
                .ok_or("Application icon is missing")?;
            TrayIconBuilder::new()
                .icon(icon)
                .tooltip("LLM-Optimise")
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "quit" => app.exit(0),
                    "show" => {
                        if let Some(window) = app.get_webview_window("lab") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        } else if let Some(window) = app.get_webview_window("launcher") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    "settings" => {
                        if let Some(window) = app.get_webview_window("launcher") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    _ => (),
                })
                .build(app)?;
            if std::env::var_os("LLM_OPTIMISE_DESKTOP_AUTOSTART").is_some() {
                let handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    let selected = defaults(handle.clone());
                    let result = start_service(
                        handle.clone(),
                        handle.state::<AppState>(),
                        Some(selected.settings.interpreter),
                        selected.settings.workspace,
                        Some(selected.settings.runtime_mode),
                    )
                    .await;
                    if let Err(error) = result {
                        eprintln!("Desktop autostart failed: {error}");
                        handle.exit(2);
                    }
                });
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Could not build LLM-Optimise desktop");
    app.run(|handle, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            let state = handle.state::<AppState>();
            if let Ok(mut service) = state.service.lock() {
                *service = None;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "llm-desktop-runtime-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn runtime(&self, relative: &str) -> PathBuf {
            let runtime = self.0.join("runtime");
            let python = runtime.join(relative);
            fs::create_dir_all(python.parent().unwrap()).unwrap();
            fs::write(&python, b"test interpreter placeholder").unwrap();
            fs::write(runtime.join("manifest.json"), serde_json::to_vec(&serde_json::json!({"format_version":1,"interpreter":relative,"python_version":"test"})).unwrap()).unwrap();
            python
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn fresh_install_uses_bundled_without_interpreter_setting() {
        let settings = Settings::default();
        assert_eq!(settings.runtime_mode, RuntimeMode::Bundled);
        assert!(settings.interpreter.is_empty());
    }

    #[test]
    fn legacy_custom_environment_remains_an_explicit_custom_choice() {
        let value = decode_settings(
            br#"{"interpreter":"/home/user/.venv/bin/python","workspace":"/home/user/lab"}"#,
        )
        .unwrap();
        assert_eq!(value.runtime_mode, RuntimeMode::Custom);
        assert_eq!(value.interpreter, "/home/user/.venv/bin/python");
        assert_eq!(value.workspace, "/home/user/lab");
        assert_eq!(value.format_version, 2);
    }

    #[test]
    fn legacy_bundled_locations_are_migrated_without_stale_paths() {
        for old in [
            "/Applications/Old.app/Contents/Resources/runtime/python/bin/python3",
            "C:\\Old\\runtime\\python\\python.exe",
        ] {
            let bytes = serde_json::to_vec(
                &serde_json::json!({"interpreter":old,"workspace":"saved-workspace"}),
            )
            .unwrap();
            let settings = decode_settings(&bytes).unwrap();
            assert_eq!(settings.runtime_mode, RuntimeMode::Bundled);
            assert!(settings.interpreter.is_empty());
            assert_eq!(settings.workspace, "saved-workspace");
        }
    }

    #[test]
    fn explicit_custom_choice_is_never_reclassified_by_its_filename() {
        let settings = decode_settings(
            br#"{"runtime_mode":"custom","interpreter":"/custom/runtime/python/bin/python3"}"#,
        )
        .unwrap();
        assert_eq!(settings.runtime_mode, RuntimeMode::Custom);
        assert!(!settings.interpreter.is_empty());
    }

    #[test]
    fn bundled_setting_never_persists_an_absolute_interpreter() {
        let settings = decode_settings(
            br#"{"runtime_mode":"bundled","interpreter":"/stale/path","workspace":"my-workspace"}"#,
        )
        .unwrap();
        let persisted = serde_json::to_value(settings).unwrap();
        assert_eq!(persisted["interpreter"], "");
        assert_eq!(persisted["workspace"], "my-workspace");
    }

    #[test]
    fn bundle_is_resolved_again_after_move_and_interpreter_upgrade() {
        let fixture = Fixture::new();
        fixture.runtime("python/bin/python3");
        let settings = Settings::default();
        let original = resolve_python(&fixture.0, &settings).unwrap();
        let moved = Fixture::new();
        fs::rename(fixture.0.join("runtime"), moved.0.join("runtime")).unwrap();
        let new_path = resolve_python(&moved.0, &settings).unwrap();
        assert_ne!(original, new_path);
        assert!(new_path.starts_with(moved.0.canonicalize().unwrap()));
        moved.runtime("python/bin/python-new");
        assert!(
            resolve_python(&moved.0, &settings)
                .unwrap()
                .ends_with("python-new")
        );
    }

    #[test]
    fn source_build_missing_runtime_does_not_search_system_path() {
        let fixture = Fixture::new();
        assert!(
            resolve_python(&fixture.0, &Settings::default())
                .unwrap_err()
                .contains("no bundled runtime")
        );
    }

    #[test]
    fn manifest_rejects_unknown_version_absolute_and_traversal_paths() {
        let fixture = Fixture::new();
        fixture.runtime("python/bin/python3");
        for (version, path) in [
            (2, "python/bin/python3"),
            (1, "../outside"),
            (1, "/outside"),
            (1, ""),
        ] {
            fs::write(
                fixture.0.join("runtime/manifest.json"),
                serde_json::to_vec(
                    &serde_json::json!({"format_version":version,"interpreter":path}),
                )
                .unwrap(),
            )
            .unwrap();
            assert!(bundled_python(&fixture.0).is_err());
        }
    }

    #[test]
    fn missing_interpreter_has_reinstall_guidance() {
        let fixture = Fixture::new();
        let python = fixture.runtime("python/bin/python3");
        fs::remove_file(python).unwrap();
        assert!(
            bundled_python(&fixture.0)
                .unwrap_err()
                .contains("no system Python is required")
        );
    }

    #[cfg(unix)]
    #[test]
    fn interpreter_symlink_must_stay_in_bundle_but_custom_venv_identity_is_preserved() {
        use std::os::unix::fs::symlink;
        let fixture = Fixture::new();
        let python = fixture.runtime("python/bin/python3");
        let external = fixture.0.join("outside-python");
        fs::write(&external, b"placeholder").unwrap();
        fs::remove_file(&python).unwrap();
        symlink(&external, &python).unwrap();
        assert!(bundled_python(&fixture.0).is_err());
        let custom = Settings {
            runtime_mode: RuntimeMode::Custom,
            interpreter: python.to_string_lossy().into(),
            ..Settings::default()
        };
        assert_eq!(resolve_python(&fixture.0, &custom).unwrap(), python);
    }
}
