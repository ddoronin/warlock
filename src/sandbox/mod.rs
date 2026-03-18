use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub mod docker;
pub mod local;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TestResult {
    pub exit_code: i64,
    pub stdout: String,
    pub stderr: String,
}

impl TestResult {
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
}

#[async_trait]
pub trait SandboxRunner: Send + Sync {
    async fn run_command(&self, repo_path: &Path, command: &str) -> Result<TestResult>;

    async fn run_tests(&self, repo_path: &Path, test_command: &str) -> Result<TestResult> {
        self.run_command(repo_path, test_command).await
    }
}
