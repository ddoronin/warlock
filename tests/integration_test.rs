use anyhow::Result;
use async_trait::async_trait;
use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use std::process::Command as ProcessCommand;
use std::sync::Arc;
use tempfile::tempdir;
use warlock::agents::coder::Coder;
use warlock::agents::planner::Planner;
use warlock::agents::reflector::Reflector;
use warlock::config::Config;
use warlock::llm::provider::{CompletionConfig, LlmProvider, Message};
use warlock::orchestrator::workflow::Orchestrator;

struct MockLlm;

#[async_trait]
impl LlmProvider for MockLlm {
    async fn complete(&self, messages: &[Message], _config: &CompletionConfig) -> Result<String> {
        let prompt = messages.last().map(|m| m.content.as_str()).unwrap_or_default();

        if prompt.contains("Return JSON array with objects") {
            return Ok(
                r#"[
                    {
                      "step": 1,
                      "task": "Fix add function implementation",
                      "target_files": ["src/lib.rs"],
                      "depends_on": []
                    }
                ]"#
                    .to_string(),
            );
        }

        if prompt.contains("Return only a valid unified diff") {
            return Ok(
                r#"diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,3 @@
 pub fn add(a: i32, b: i32) -> i32 {
-    0
+    a + b
 }
"#
                .to_string(),
            );
        }

        Ok("ESCALATE unexpected prompt".to_string())
    }

    fn provider_name(&self) -> &'static str {
        "mock"
    }
}

#[tokio::test]
async fn full_plan_code_test_loop_opt_in() {
    if std::env::var("WARLOCK_RUN_INTEGRATION").ok().as_deref() != Some("1") {
        return;
    }

    let tmp = tempdir().expect("tempdir");
    let repo = tmp.path();
    setup_fixture_repo(repo);

    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("warlock.toml");
    let mut config = Config::load(config_path).expect("load config");
    config.llm.provider = "mock".to_string();
    config.agent.max_reflection_attempts = 2;

    let llm = Arc::new(MockLlm);
    let completion = CompletionConfig {
        model: "mock".to_string(),
        temperature: 0.0,
        max_tokens: 512,
        json_mode: false,
    };

    let planner = Planner::new(llm.clone(), completion.clone());
    let coder = Coder::new(llm.clone(), completion.clone());
    let reflector = Reflector::new(llm, completion);

    let mut orchestrator = Orchestrator::new(config, repo.to_path_buf(), planner, coder, reflector)
        .await
        .expect("create orchestrator");

    let report = orchestrator.solve("Fix add function").await.expect("solve should run");
    assert!(report.overall_success, "report should indicate success");
    assert!(report
        .step_results
        .iter()
        .any(|s| matches!(s.status, warlock::orchestrator::workflow::StepStatus::Succeeded)));
}

fn setup_fixture_repo(repo: &Path) {
    run(repo, "git init");
    run(repo, "git config user.email test@example.com");
    run(repo, "git config user.name Tester");

    fs::write(
        repo.join("Cargo.toml"),
        r#"[package]
name = "fixture"
version = "0.1.0"
edition = "2021"
"#,
    )
    .expect("write Cargo.toml");

    fs::create_dir_all(repo.join("src")).expect("create src");
    fs::write(
        repo.join("src/lib.rs"),
        r#"pub fn add(a: i32, b: i32) -> i32 {
    0
}
"#,
    )
    .expect("write lib");

    fs::create_dir_all(repo.join("tests")).expect("create tests");
    fs::write(
        repo.join("tests/add_test.rs"),
        r#"use fixture::add;

#[test]
fn add_works() {
    assert_eq!(add(2, 3), 5);
}
"#,
    )
    .expect("write test");

    run(repo, "git add .");
    run(repo, "git commit -m init --quiet");
}

fn run(repo: &Path, shell: &str) {
    let status = ProcessCommand::new("sh")
        .arg("-lc")
        .arg(shell)
        .current_dir(repo)
        .status()
        .expect("spawn shell command");
    assert!(status.success(), "command failed: {shell}");
}

#[test]
fn cli_help_lists_search_command() {
    let mut cmd = Command::cargo_bin("warlock").expect("binary should exist");
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("search"));
}

#[test]
fn search_requires_repo_flag() {
    let mut cmd = Command::cargo_bin("warlock").expect("binary should exist");
    cmd.args(["search", "find parser logic"]) 
        .assert()
        .failure()
        .stderr(predicate::str::contains("--repo"));
}
