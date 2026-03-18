use anyhow::{Context, Result};
use async_trait::async_trait;
use bollard::container::{
    Config as ContainerConfig, CreateContainerOptions, LogsOptions, RemoveContainerOptions,
    StartContainerOptions, WaitContainerOptions,
};
use bollard::models::HostConfig;
use bollard::Docker;
use futures::StreamExt;
use std::path::Path;
use tokio::time::{timeout, Duration};
use uuid::Uuid;

use crate::config::SandboxConfig;
use crate::sandbox::{SandboxRunner, TestResult};

pub struct Sandbox {
    docker: Docker,
    image: String,
    timeout_secs: u64,
    cpu_limit: u32,
    memory_limit_mb: u64,
    network_enabled: bool,
}

impl Sandbox {
    pub fn new(config: &SandboxConfig) -> Result<Self> {
        let docker = Docker::connect_with_local_defaults()
            .context("failed to connect to local Docker daemon")?;
        Ok(Self {
            docker,
            image: config.image.clone(),
            timeout_secs: config.timeout_secs,
            cpu_limit: config.cpu_limit,
            memory_limit_mb: config.memory_limit_mb,
            network_enabled: config.network_enabled,
        })
    }

    async fn run_command_impl(&self, repo_path: &Path, command: &str) -> Result<TestResult> {
        let container_name = format!("warlock-sandbox-{}", Uuid::new_v4());
        let mount = format!("{}:/workspace:ro", repo_path.display());

        let host_config = HostConfig {
            binds: Some(vec![mount]),
            network_mode: Some(if self.network_enabled {
                "bridge".to_string()
            } else {
                "none".to_string()
            }),
            memory: Some((self.memory_limit_mb * 1024 * 1024) as i64),
            nano_cpus: Some((self.cpu_limit as i64) * 1_000_000_000),
            ..Default::default()
        };

        let create = self
			.docker
			.create_container(
				Some(CreateContainerOptions {
					name: container_name.clone(),
					platform: None,
				}),
				ContainerConfig {
					image: Some(self.image.clone()),
					cmd: Some(vec![
						"sh".to_string(),
						"-lc".to_string(),
						format!(
							"rm -rf /tmp/workspace && cp -R /workspace /tmp/workspace && chmod -R u+w /tmp/workspace && cd /tmp/workspace && {command}"
						),
					]),
					host_config: Some(host_config),
					attach_stdout: Some(true),
					attach_stderr: Some(true),
					tty: Some(false),
					..Default::default()
				},
			)
			.await
			.context("failed to create sandbox container")?;

        self.docker
            .start_container(&create.id, None::<StartContainerOptions<String>>)
            .await
            .context("failed to start sandbox container")?;

        let log_fut = async {
            let mut stdout = String::new();
            let mut stderr = String::new();
            let mut logs = self.docker.logs(
                &create.id,
                Some(LogsOptions::<String> {
                    follow: true,
                    stdout: true,
                    stderr: true,
                    timestamps: false,
                    tail: "all".to_string(),
                    ..Default::default()
                }),
            );

            while let Some(item) = logs.next().await {
                match item {
                    Ok(bollard::container::LogOutput::StdOut { message }) => {
                        stdout.push_str(&String::from_utf8_lossy(&message));
                    }
                    Ok(bollard::container::LogOutput::StdErr { message }) => {
                        stderr.push_str(&String::from_utf8_lossy(&message));
                    }
                    Ok(bollard::container::LogOutput::Console { message }) => {
                        stdout.push_str(&String::from_utf8_lossy(&message));
                    }
                    _ => {}
                }
            }

            Ok::<(String, String), anyhow::Error>((stdout, stderr))
        };

        let wait_fut = async {
            let mut wait_stream = self
                .docker
                .wait_container(&create.id, None::<WaitContainerOptions<String>>);
            match wait_stream.next().await {
                Some(Ok(result)) => Ok(result.status_code),
                Some(Err(e)) => Err(anyhow::anyhow!(e).context("wait_container failed")),
                None => Ok(-1),
            }
        };

        let timed = timeout(Duration::from_secs(self.timeout_secs), async {
            tokio::try_join!(log_fut, wait_fut)
        })
        .await;

        let result = match timed {
            Ok(Ok(((stdout, stderr), exit_code))) => TestResult {
                exit_code,
                stdout,
                stderr,
            },
            Ok(Err(e)) => TestResult {
                exit_code: -1,
                stdout: String::new(),
                stderr: e.to_string(),
            },
            Err(_) => TestResult {
                exit_code: -1,
                stdout: String::new(),
                stderr: format!("test run timed out after {} seconds", self.timeout_secs),
            },
        };

        let _ = self
            .docker
            .remove_container(
                &create.id,
                Some(RemoveContainerOptions {
                    force: true,
                    ..Default::default()
                }),
            )
            .await;

        Ok(result)
    }
}

#[async_trait]
impl SandboxRunner for Sandbox {
    async fn run_command(&self, repo_path: &Path, command: &str) -> Result<TestResult> {
        self.run_command_impl(repo_path, command).await
    }

    async fn run_tests(&self, repo_path: &Path, test_command: &str) -> Result<TestResult> {
        self.run_command_impl(repo_path, test_command).await
    }
}
