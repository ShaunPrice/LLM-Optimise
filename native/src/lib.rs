//! Owned process groups/jobs. Only the worker created by this guard is controlled.
use std::{
    collections::{HashMap, HashSet},
    io,
    process::{Child, Command, ExitStatus},
    thread,
    time::{Duration, Instant},
};
use sysinfo::{Pid, System};
#[cfg(unix)]
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate};

pub struct OwnedProcess {
    pub child: Child,
    finished: bool,
    known: HashMap<Pid, u64>,
    pub group_signal_fallback: bool,
    #[cfg(windows)]
    job: windows_sys::Win32::Foundation::HANDLE,
}

// Windows HANDLE belongs solely to this guard; no borrowed external handles.
#[cfg(windows)]
unsafe impl Send for OwnedProcess {}

impl OwnedProcess {
    pub fn spawn(command: &mut Command) -> io::Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(
                windows_sys::Win32::System::Threading::CREATE_NEW_PROCESS_GROUP
                    | windows_sys::Win32::System::Threading::CREATE_SUSPENDED,
            );
        }
        let mut child = command.spawn()?;
        #[cfg(windows)]
        let job = unsafe {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::{Foundation::CloseHandle, System::JobObjects::*};
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                let _ = child.kill();
                let _ = child.wait();
                return Err(io::Error::last_os_error());
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as _,
                std::mem::size_of_val(&info) as u32,
            ) == 0
                || AssignProcessToJobObject(job, child.as_raw_handle() as _) == 0
            {
                let error = io::Error::last_os_error();
                let _ = child.kill();
                let _ = child.wait();
                CloseHandle(job);
                return Err(error);
            }
            // Assign the suspended process to the job before it can create descendants.
            use windows_sys::Win32::System::{Diagnostics::ToolHelp::*, Threading::*};
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            let mut resumed = false;
            if snapshot != windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
                let mut entry: THREADENTRY32 = std::mem::zeroed();
                entry.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
                let mut found = Thread32First(snapshot, &mut entry);
                while found != 0 {
                    if entry.th32OwnerProcessID == child.id() {
                        let thread = OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID);
                        if !thread.is_null() {
                            resumed = ResumeThread(thread) != u32::MAX;
                            CloseHandle(thread);
                        }
                        break;
                    }
                    found = Thread32Next(snapshot, &mut entry);
                }
                CloseHandle(snapshot);
            }
            if !resumed {
                let error =
                    io::Error::other("could not resume the owned worker after job assignment");
                TerminateJobObject(job, 137);
                let _ = child.wait();
                CloseHandle(job);
                return Err(error);
            }
            job
        };
        #[cfg(not(windows))]
        let _ = &mut child;
        Ok(Self {
            child,
            finished: false,
            known: HashMap::new(),
            group_signal_fallback: false,
            #[cfg(windows)]
            job,
        })
    }

    pub fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn record_tree(&mut self, system: &System) {
        let mut owned = HashSet::from([Pid::from_u32(self.id())]);
        #[cfg(unix)]
        for pid in system.processes().keys() {
            // The unreaped group leader retains this group id; this also finds
            // grandchildren reparented after an intermediate worker exits.
            if unsafe { libc::getpgid(pid.as_u32() as i32) } == self.id() as i32 {
                owned.insert(*pid);
            }
        }
        loop {
            let count = owned.len();
            for (pid, process) in system.processes() {
                if process
                    .parent()
                    .is_some_and(|parent| owned.contains(&parent))
                {
                    owned.insert(*pid);
                }
            }
            if count == owned.len() {
                break;
            }
        }
        for pid in owned {
            if let Some(process) = system.process(pid) {
                self.known.insert(pid, process.start_time());
            }
        }
    }

    /// Detect exit without reaping Unix leader, retaining its PID until group cleanup.
    pub fn exited(&mut self) -> io::Result<bool> {
        #[cfg(unix)]
        unsafe {
            let mut info: libc::siginfo_t = std::mem::zeroed();
            if libc::waitid(
                libc::P_PID,
                self.id(),
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            ) != 0
            {
                return Err(io::Error::last_os_error());
            }
            #[cfg(target_os = "macos")]
            let pid = info.si_pid;
            #[cfg(not(target_os = "macos"))]
            let pid = info.si_pid();
            Ok(pid != 0)
        }
        #[cfg(windows)]
        {
            Ok(self.child.try_wait()?.is_some())
        }
    }

    pub fn signal(&mut self, force: bool) {
        #[cfg(unix)]
        unsafe {
            let result = libc::kill(
                -(self.id() as i32),
                if force { libc::SIGKILL } else { libc::SIGTERM },
            );
            if result != 0 {
                self.group_signal_fallback = true;
                // Some macOS execution sandboxes forbid negative-PID signals.
                // Fall back to only recorded, identity-checked owned processes.
                let mut system = System::new();
                system.refresh_processes_specifics(
                    ProcessesToUpdate::All,
                    true,
                    ProcessRefreshKind::nothing(),
                );
                self.record_tree(&system);
                for (pid, birth) in &self.known {
                    if let Some(process) = system.process(*pid)
                        && process.start_time() == *birth
                    {
                        libc::kill(
                            pid.as_u32() as i32,
                            if force { libc::SIGKILL } else { libc::SIGTERM },
                        );
                    }
                }
                // The unreaped Child still owns this leader PID, including zombies.
                libc::kill(
                    self.id() as i32,
                    if force { libc::SIGKILL } else { libc::SIGTERM },
                );
            }
        }
        #[cfg(windows)]
        unsafe {
            if force {
                windows_sys::Win32::System::JobObjects::TerminateJobObject(self.job, 137);
            } else {
                windows_sys::Win32::System::Console::GenerateConsoleCtrlEvent(
                    windows_sys::Win32::System::Console::CTRL_BREAK_EVENT,
                    self.id(),
                );
            }
        }
    }

    pub fn finish(&mut self, grace: Duration) -> io::Result<ExitStatus> {
        if self.finished {
            return self.child.wait();
        }
        self.signal(false);
        let deadline = Instant::now() + grace;
        while Instant::now() < deadline {
            if self.exited().unwrap_or(true) {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        // Kill remaining group/job members even when the original worker exited.
        self.signal(true);
        let result = self.child.wait();
        self.finished = true;
        result
    }
}

impl Drop for OwnedProcess {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finish(Duration::from_millis(100));
        }
        #[cfg(windows)]
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.job);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owns_and_reaps_worker() {
        let mut command = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", "exit", "0"]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", "exit 0"]);
            c
        };
        let mut worker = OwnedProcess::spawn(&mut command).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while !worker.exited().unwrap() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(worker.exited().unwrap());
        assert!(worker.finish(Duration::ZERO).unwrap().success());
    }
}
