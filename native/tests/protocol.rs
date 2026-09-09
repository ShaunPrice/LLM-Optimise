#![cfg(unix)]
use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Stdio},
};

fn run(config: Value) -> (i32, Vec<Value>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_llm-supervisor"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.take().unwrap(), "{config}").unwrap();
    let result = child.wait_with_output().unwrap();
    let events = String::from_utf8(result.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    (result.status.code().unwrap(), events)
}
#[test]
fn preserves_exit_code_and_child_signal() {
    for (command, expected, signal) in [("exit 7", 7, None), ("kill -TERM $$", 143, Some(15))] {
        let (code, events) =
            run(json!({"command":["/bin/sh","-c",command],"cancel_on_stdin_eof":false}));
        assert_eq!(code, expected);
        assert_eq!(events.last().unwrap()["reason"], "exited");
        assert_eq!(events.last().unwrap()["worker_signal"], json!(signal));
    }
}
#[test]
fn enforces_timeout_and_sampled_rss() {
    for (extra, expected, reason) in [
        (json!({"timeout_ms":100}), 124, "timeout"),
        (json!({"rss_limit_bytes":1}), 125, "rss_limit"),
    ] {
        let mut config = json!({"command":["/bin/sh","-c","sleep 10"],"sample_ms":10,"grace_ms":20,"cancel_on_stdin_eof":false});
        config
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let (code, events) = run(config);
        assert_eq!(code, expected);
        assert_eq!(events.last().unwrap()["reason"], reason);
    }
}
#[test]
fn controller_eof_cancels_owned_worker() {
    let (code, events) = run(json!({"command":["/bin/sh","-c","sleep 10"],"grace_ms":20}));
    assert_eq!(code, 130);
    assert_eq!(events.last().unwrap()["reason"], "controller_closed");
}
#[test]
fn invalid_config_does_not_launch() {
    let (code, events) = run(json!({"command":[],"timeout_ms":0}));
    assert_eq!(code, 2);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["event"], "error");
}
