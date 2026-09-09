# Optional Rust process supervisor

The supervisor launches one owned worker, measures its process-tree RSS, handles cancellation and deadlines, and reaps it. It does not implement model inference, GPU kernels, or an OS memory sandbox. The Python application remains usable without Rust; `python_supervisor.py` provides the same JSONL control protocol for the matched-workload reference tests.

```bash
cargo test --manifest-path native/Cargo.toml
cargo build --release --locked --manifest-path native/Cargo.toml
```

Rust 1.95+ is required by the locked `sysinfo` dependency. The executable is `native/target/release/llm-supervisor` (`.exe` on Windows). Build each OS/architecture with its native toolchain. macOS arm64 and native Windows x64/MSVC were compiled and exercised. Linux runtime validation is a separate gate.

## Controller protocol

Launch the binary with stdin/stdout pipes. Send one JSON configuration line; keep stdin open until the worker should stop. No shell is invoked and arguments are not interpolated. Worker stdout/stderr go into `log_path`; supervisor stdout contains only JSON events.

```json
{"command":["/absolute/path/to/llama-server","--model","/path/model.gguf"],"cwd":"/path/workspace","log_path":"/path/worker.log","rss_limit_bytes":4294967296,"timeout_ms":60000,"sample_ms":50,"grace_ms":500,"cancel_on_stdin_eof":true}
```

Optional `cwd`, `log_path`, `rss_limit_bytes` and `timeout_ms` may be omitted. `sample_ms` defaults to 50 and must be 5–10,000; `grace_ms` defaults to 500 and cannot exceed 10,000. Memory and timeout limits must be positive if present. The configuration is limited to 1 MiB and 256 arguments. Credentials are not included in events; avoid placing secrets in the command itself.

Events:

- `started`: worker `pid`, `elapsed_ms`, `supervisor`.
- `sample`: `rss_bytes`, `processes`, `elapsed_ms`.
- `finished`: `reason`, `worker_exit_code`, `worker_signal`, `exit_code`, `rss_peak_bytes`, `samples`, `sample_ms`, `memory_scope`, `group_signal_fallback`.
- `error`: a configuration, launch or supervision error. Treat uncertain startup as a failure requiring cleanup; do not launch a duplicate fallback worker.

Send `{"command":"cancel"}` followed by a newline to cancel. EOF cancels by default. `--config path.json` reads configuration from a file instead; set `cancel_on_stdin_eof:false` for an unattended command with no controller pipe. An ordinary worker exit preserves its code; Unix worker termination signals are recorded and returned as `128 + signal`. Supervisor cancellation/interrupt returns 130, deadline 124, RSS breach 125, and protocol/setup failure 2.

## Process ownership and limits

On Unix a new process group is created. Exit detection retains the unreaped leader until group cleanup to avoid reusing its PID/group identifier. If the host execution environment rejects a group signal, cleanup falls back to recorded process identities and matching owned group members. Both active descendants and descendants left behind by an exiting worker are checked in the live tests. A process deliberately escaping the group is outside this containment model; this is not a security boundary for adversarial code.

On Windows the worker starts suspended, is assigned to a job with `KILL_ON_JOB_CLOSE`, then resumes. This prevents a child-spawn gap before job assignment. Graceful termination uses a console break when available; the job is forcibly terminated after the grace interval. Native Omen execution passed exit42, RSS termination, deadline termination, cancellation with a child, and orphan-child cleanup; see [Windows validation](validation/windows-x64-supervisor-validation.json). `windows_validate.py` reproduces those checks using standard-library Python and the native release binary.

RSS enforcement is sampled and cooperative with the supervisor's scheduler: allocation spikes between samples can be missed. GPU allocations are not measured by this component. The application lifecycle manager handles its NVIDIA telemetry separately. Keep Docker/OS restrictions for generated untrusted code.

## Actual comparison

Run from the repository's Python environment, with `psutil` installed:

```bash
python native/profile.py
```

The harness runs the same idle, RSS-limit, deadline, cancellation, orphan-descendant and exit-code workloads through Rust and Python, with three repeats and alternating order. It separately samples the supervisor process's RSS and CPU time. Worker memory is kept in its own result field and is not attributed to the supervisor. Results are in `native/profile-local/results.json`; a recorded macOS result is in [validation/macos-arm64-supervisor-profile.json](validation/macos-arm64-supervisor-profile.json).

The measured benefit is reduced supervisor startup time and resident memory on this Mac. Cancellation latency is similar and does not consistently favour Rust. No LLM tokens/second, GPU-memory reduction, total-app memory reduction, or general cross-platform performance improvement is inferred from this small overhead test. The Python path remains a valid choice when another executable and build toolchain would add more operational cost than these savings justify.

The final Mac run recorded idle supervisor RSS medians of **7.83 MiB Rust / 18.17 MiB Python**, startup medians of **4.74 ms / 28.43 ms**, and cancellation medians of **44.94 ms / 47.77 ms**. This refreshed run includes the additional orphan-child cleanup check. These measurements concern the supervisor process alone.

Primary implementation references: [sysinfo process sampling](https://docs.rs/sysinfo/0.39.6/sysinfo/), [Windows job objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).
