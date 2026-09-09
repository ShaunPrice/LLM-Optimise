use llm_supervisor::OwnedProcess;
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::HashSet,
    fs::File,
    io::{self, BufRead, Write},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    command: Vec<String>,
    cwd: Option<String>,
    log_path: Option<String>,
    rss_limit_bytes: Option<u64>,
    timeout_ms: Option<u64>,
    #[serde(default = "sample_default")]
    sample_ms: u64,
    #[serde(default = "grace_default")]
    grace_ms: u64,
    #[serde(default = "true_default")]
    cancel_on_stdin_eof: bool,
}
fn sample_default() -> u64 {
    50
}
fn grace_default() -> u64 {
    500
}
fn true_default() -> bool {
    true
}
fn emit(value: serde_json::Value) {
    println!("{value}");
    let _ = io::stdout().flush();
}

fn run() -> Result<i32, Box<dyn std::error::Error>> {
    let mut input = io::BufReader::new(io::stdin());
    let mut args = std::env::args().skip(1);
    let raw =
        match args.next().as_deref() {
            Some("--version") => {
                println!("llm-supervisor 0.1.0");
                return Ok(0);
            }
            Some("--config") => {
                std::fs::read_to_string(args.next().ok_or("--config requires a path")?)?
            }
            None => {
                let mut line = String::new();
                input.read_line(&mut line)?;
                line
            }
            _ => return Err(
                "usage: llm-supervisor [--config path]; otherwise first stdin line is JSON config"
                    .into(),
            ),
        };
    if raw.len() > 1024 * 1024 {
        return Err("configuration exceeds 1 MiB".into());
    }
    let config: Config = serde_json::from_str(&raw)?;
    if config.command.is_empty()
        || config.command.len() > 256
        || config.command.iter().any(|s| s.contains('\0'))
        || !(5..=10_000).contains(&config.sample_ms)
        || config.grace_ms > 10_000
        || config.timeout_ms == Some(0)
        || config.rss_limit_bytes == Some(0)
    {
        return Err("invalid command or resource limits".into());
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    let signal_flag = cancelled.clone();
    ctrlc::set_handler(move || {
        signal_flag.store(true, Ordering::SeqCst);
    })?;
    let (sender, receiver) = mpsc::channel();
    let eof_cancels = config.cancel_on_stdin_eof;
    thread::spawn(move || {
        let mut line = String::new();
        loop {
            line.clear();
            match input.read_line(&mut line) {
                Ok(0) => {
                    if eof_cancels {
                        let _ = sender.send("controller_closed");
                    }
                    break;
                }
                Ok(_) => {
                    if line.trim() == "cancel"
                        || serde_json::from_str::<serde_json::Value>(&line)
                            .is_ok_and(|v| v["command"] == "cancel")
                    {
                        let _ = sender.send("cancelled");
                        break;
                    }
                }
                Err(_) => {
                    let _ = sender.send("controller_error");
                    break;
                }
            }
        }
    });
    let started = Instant::now();
    let mut command = Command::new(&config.command[0]);
    command.args(&config.command[1..]).stdin(Stdio::null());
    if let Some(cwd) = config.cwd {
        command.current_dir(cwd);
    }
    if let Some(path) = config.log_path {
        let file = File::create(path)?;
        command.stdout(file.try_clone()?).stderr(file);
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let mut worker = OwnedProcess::spawn(&mut command)?;
    let root = Pid::from_u32(worker.id());
    emit(
        json!({"event":"started","pid":worker.id(),"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"supervisor":"rust"}),
    );
    let mut system = System::new();
    let mut peak = 0_u64;
    let mut samples = 0;
    let reason;
    loop {
        if worker.exited()? {
            reason = "exited";
            break;
        }
        if cancelled.load(Ordering::SeqCst) {
            reason = "signal";
            break;
        }
        if let Ok(value) = receiver.try_recv() {
            reason = value;
            break;
        }
        if config
            .timeout_ms
            .is_some_and(|limit| started.elapsed().as_millis() >= limit as u128)
        {
            reason = "timeout";
            break;
        }
        system.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        worker.record_tree(&system);
        let mut owned = HashSet::from([root]);
        loop {
            let before = owned.len();
            for (pid, process) in system.processes() {
                if process
                    .parent()
                    .is_some_and(|parent| owned.contains(&parent))
                {
                    owned.insert(*pid);
                }
            }
            if owned.len() == before {
                break;
            }
        }
        let rss: u64 = owned
            .iter()
            .filter_map(|pid| system.process(*pid))
            .map(|p| p.memory())
            .sum();
        peak = peak.max(rss);
        samples += 1;
        emit(
            json!({"event":"sample","elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"rss_bytes":rss,"processes":owned.len()}),
        );
        if config.rss_limit_bytes.is_some_and(|limit| rss > limit) {
            reason = "rss_limit";
            break;
        }
        thread::sleep(Duration::from_millis(config.sample_ms));
    }
    let status = worker.finish(Duration::from_millis(config.grace_ms))?;
    #[cfg(unix)]
    let worker_signal = {
        use std::os::unix::process::ExitStatusExt;
        status.signal()
    };
    #[cfg(not(unix))]
    let worker_signal: Option<i32> = None;
    let code = if reason == "exited" {
        status.code().unwrap_or(128 + worker_signal.unwrap_or(1))
    } else if reason == "timeout" {
        124
    } else if reason == "rss_limit" {
        125
    } else {
        130
    };
    emit(
        json!({"event":"finished","reason":reason,"worker_exit_code":status.code(),"worker_signal":worker_signal,"exit_code":code,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"rss_peak_bytes":peak,"samples":samples,"sample_ms":config.sample_ms,"group_signal_fallback":worker.group_signal_fallback,"memory_scope":"sampled owned process tree RSS; excludes GPU; short peaks may be missed"}),
    );
    Ok(code)
}
fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(error) => {
            emit(json!({"event":"error","error":error.to_string()}));
            2
        }
    };
    std::process::exit(code);
}
