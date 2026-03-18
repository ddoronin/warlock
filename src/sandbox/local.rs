use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

use crate::config::SandboxConfig;
use crate::sandbox::{SandboxRunner, TestResult};

pub struct LocalSandbox {
    timeout_secs: u64,
}

impl LocalSandbox {
    pub fn new(config: &SandboxConfig) -> Self {
        Self {
            timeout_secs: config.timeout_secs,
        }
    }
}

#[async_trait]
impl SandboxRunner for LocalSandbox {
    async fn run_command(&self, repo_path: &Path, command: &str) -> Result<TestResult> {
        let mut cmd = Command::new("sh");
        cmd.arg("-lc").arg(command).current_dir(repo_path);

        let timed = timeout(Duration::from_secs(self.timeout_secs), cmd.output()).await;
        match timed {
            Ok(Ok(output)) => Ok(TestResult {
                exit_code: output.status.code().unwrap_or(-1) as i64,
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            }),
            Ok(Err(err)) => Err(anyhow::anyhow!(
                "failed to run local command: {command}: {err}"
            )),
            Err(_) => Ok(TestResult {
                exit_code: -1,
                stdout: String::new(),
                stderr: format!("command timed out after {} seconds", self.timeout_secs),
            }),
        }
    }
}
