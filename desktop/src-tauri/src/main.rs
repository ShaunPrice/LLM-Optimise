#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use llm_supervisor::OwnedProcess;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
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

#[derive(Default, Clone, Serialize, Deserialize)]
struct Settings {
    interpreter: String,
    workspace: String,
}
struct Service {
    process: OwnedProcess,
    input: Option<ChildStdin>,
    url: String,
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
fn defaults(app: tauri::AppHandle) -> Settings {
    let mut value = settings_path(&app)
        .ok()
        .and_then(|p| fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<Settings>(&b).ok())
        .unwrap_or_default();
    if let Ok(python) = std::env::var("LLM_OPTIMISE_PYTHON") {
        value.interpreter = python;
    }
    if let Ok(workspace) = std::env::var("LLM_OPTIMISE_WORKSPACE") {
        value.workspace = workspace;
    }
    if value.workspace.is_empty() {
        value.workspace = app
            .path()
            .document_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("LLM-Optimise")
            .to_string_lossy()
            .into();
    }
    value
}

fn launch(app: &tauri::AppHandle, interpreter: &str, workspace: &str) -> Result<Service, String> {
    let python = Path::new(interpreter);
    if !python.is_absolute() || !python.is_file() {
        return Err("Select an existing Python executable using its absolute path.".into());
    }
    let workspace = Path::new(workspace);
    if !workspace.is_absolute() {
        return Err("The workspace must be an absolute directory path.".into());
    }
    fs::create_dir_all(workspace).map_err(|e| format!("Cannot create workspace: {e}"))?;
    let script = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("service.py");
    let script = if script.is_file() {
        script
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/service.py")
    };
    let logs = workspace.join(".llm-optimise").join("desktop");
    fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
    let log = fs::File::create(logs.join("service.log")).map_err(|e| e.to_string())?;
    let mut command = Command::new(python);
    command
        .args(["-I", "-u"])
        .arg(script)
        .arg("--workspace")
        .arg(workspace)
        .current_dir(workspace)
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
        let result = BufReader::new(output).read_line(&mut line).map(|_| line);
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
    {
        return Err("Service must use its own loopback-only HTTP endpoint.".into());
    }
    let input = process.child.stdin.take();
    Ok(Service {
        process,
        input,
        url,
    })
}

fn show_lab(app: &tauri::AppHandle, url: &str) -> Result<(), String> {
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
    let window=WebviewWindowBuilder::new(app,"lab",WebviewUrl::External(url.parse().map_err(|_|"Invalid URL")?))
        .title("LLM-Optimise · Laboratory").inner_size(1320.0,880.0).min_inner_size(900.0,620.0)
        .on_navigation(move |target|target.as_str()==origin || target.as_str().starts_with(&(origin.clone()+"/")))
        .on_page_load(move |_window,payload| {
            if payload.event()==tauri::webview::PageLoadEvent::Finished
                && let Some(path)=std::env::var_os("LLM_OPTIMISE_DESKTOP_TEST_REPORT") {
                    let _=fs::write(path,serde_json::to_vec_pretty(&serde_json::json!({"event":"native_webview_loaded","url":payload.url().as_str(),"tauri":2,"sidecar":"selected Python interpreter"})).unwrap());
                    if std::env::var_os("LLM_OPTIMISE_DESKTOP_SMOKE_EXIT").is_some() {let app=handle.clone();thread::spawn(move || {thread::sleep(Duration::from_millis(500));app.exit(0);});}
            }
        }).build().map_err(|e|e.to_string())?;
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
    interpreter: String,
    workspace: String,
) -> Result<serde_json::Value, String> {
    let mut service = state
        .service
        .lock()
        .map_err(|_| "Service state unavailable")?;
    if let Some(running) = service.as_mut() {
        if !running.process.exited().unwrap_or(true) {
            show_lab(&app, &running.url)?;
            return Ok(serde_json::json!({"url":running.url}));
        }
        *service = None;
    }
    let running = launch(&app, &interpreter, &workspace)?;
    let url = running.url.clone();
    *service = Some(running);
    if let Err(error) = show_lab(&app, &url) {
        *service = None;
        return Err(error);
    }
    let path = settings_path(&app)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(&Settings {
            interpreter,
            workspace,
        })
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
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
                        selected.interpreter,
                        selected.workspace,
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
