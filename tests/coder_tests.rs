use async_trait::async_trait;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use tempfile::tempdir;
use warlock::agents::coder::{parse_unified_diff, Coder};
use warlock::agents::planner::PlanStep;
use warlock::llm::provider::{CompletionConfig, LlmProvider, Message};
use warlock::patch::apply::apply_patch;
use warlock::patch::revert::revert_applied_patches;

#[test]
fn parses_unified_diff_blocks() {
    let diff = r#"diff --git a/src/lib.rs b/src/lib.rs
index 1111111..2222222 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1 +1 @@
-pub fn val() -> i32 { 1 }
+pub fn val() -> i32 { 2 }
"#;

    let patches = parse_unified_diff(diff).expect("diff should parse");
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].file, Path::new("src/lib.rs"));
    assert!(!patches[0].is_new_file);
}

#[test]
fn parses_plain_unified_diff_without_git_header() {
    let diff = r#"--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1 +1 @@
-pub fn val() -> i32 { 1 }
+pub fn val() -> i32 { 2 }
"#;

    let patches = parse_unified_diff(diff).expect("plain unified diff should parse");
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].file, Path::new("src/lib.rs"));
}

#[test]
fn parses_markdown_wrapped_diff() {
    let diff = r#"Here is the patch:

```diff
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1 +1 @@
-pub fn val() -> i32 { 1 }
+pub fn val() -> i32 { 2 }
```
"#;

    let patches = parse_unified_diff(diff).expect("markdown wrapped diff should parse");
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].file, Path::new("src/lib.rs"));
}

#[test]
fn parses_begin_patch_update_file_format() {
    let diff = r#"*** Begin Patch
*** Update File: src/lib.rs
@@ -1 +1 @@
-pub fn val() -> i32 { 1 }
+pub fn val() -> i32 { 2 }
*** End Patch
"#;

    let patches = parse_unified_diff(diff).expect("begin patch format should be converted");
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].file, Path::new("src/lib.rs"));
    assert!(!patches[0].is_new_file);
    assert!(patches[0]
        .diff
        .contains("diff --git a/src/lib.rs b/src/lib.rs"));
}

#[test]
fn parses_begin_patch_add_file_format() {
    let diff = r#"*** Begin Patch
*** Add File: src/new_file.rs
@@ -0,0 +1 @@
+pub const VALUE: i32 = 42;
*** End Patch
"#;

    let patches = parse_unified_diff(diff).expect("begin patch add file should be converted");
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].file, Path::new("src/new_file.rs"));
    assert!(patches[0].is_new_file);
    assert!(patches[0].diff.contains("--- /dev/null"));
    assert!(patches[0].diff.contains("+++ b/src/new_file.rs"));
}

#[test]
fn parses_begin_patch_add_file_without_hunk_header() {
    let diff = r#"*** Begin Patch
*** Add File: docs/module_summaries.md
+# Module summaries
+
+This is a generated summary document.
*** End Patch
"#;

    let patches = parse_unified_diff(diff).expect("begin patch add file should synthesize hunks");
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].file, Path::new("docs/module_summaries.md"));
    assert!(patches[0].is_new_file);
    assert!(patches[0].diff.contains("@@ -0,0 +1,3 @@"));
}

#[test]
fn begin_patch_update_with_plain_context_applies_successfully() {
    let tmp = tempdir().expect("tempdir");
    let repo = tmp.path();

    run(repo, "git init");
    run(repo, "git config user.email test@example.com");
    run(repo, "git config user.name Tester");

    fs::create_dir_all(repo.join("src/indexing")).expect("create nested dir");
    fs::write(
        repo.join("src/indexing/mod.rs"),
        "pub fn build_module_summaries() -> Vec<String> {\n    let mut out = vec![];\n    out.push(\"ok\".to_string());\n    out\n}\n",
    )
    .expect("write baseline file");
    run(repo, "git add src/indexing/mod.rs");
    run(repo, "git commit -m init --quiet");

    let begin_patch = r#"*** Begin Patch
*** Update File: src/indexing/mod.rs
@@
 pub fn build_module_summaries() -> Vec<String> {
     let mut out = vec![];
+    // inserted doc hint
     out.push("ok".to_string());
     out
 }
*** End Patch
"#;

    let patches = parse_unified_diff(begin_patch).expect("begin patch should parse");
    assert_eq!(patches.len(), 1);

    apply_patch(repo, &patches[0].diff).expect("converted patch should apply");

    let updated = fs::read_to_string(repo.join("src/indexing/mod.rs")).expect("read updated");
    assert!(updated.contains("// inserted doc hint"));
}

#[test]
fn apply_and_revert_patch_in_temp_repo() {
    let tmp = tempdir().expect("tempdir");
    let repo = tmp.path();

    run(repo, "git init");
    run(repo, "git config user.email test@example.com");
    run(repo, "git config user.name Tester");

    fs::write(repo.join("hello.txt"), "hello\n").expect("write file");
    run(repo, "git add hello.txt");
    run(repo, "git commit -m init --quiet");

    let diff = r#"diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-hello
+world
"#;

    apply_patch(repo, diff).expect("patch should apply");
    let changed = fs::read_to_string(repo.join("hello.txt")).expect("read changed");
    assert_eq!(changed, "world\n");

    revert_applied_patches(repo).expect("revert should succeed");
    let reverted = fs::read_to_string(repo.join("hello.txt")).expect("read reverted");
    assert_eq!(reverted, "hello\n");
}

struct ScriptedLlm {
    outputs: Mutex<Vec<String>>,
}

#[async_trait]
impl LlmProvider for ScriptedLlm {
    async fn complete(
        &self,
        _messages: &[Message],
        _config: &CompletionConfig,
    ) -> anyhow::Result<String> {
        let mut guard = self.outputs.lock().expect("lock outputs");
        if guard.is_empty() {
            return Err(anyhow::anyhow!("no scripted output available"));
        }
        Ok(guard.remove(0))
    }

    fn provider_name(&self) -> &'static str {
        "scripted"
    }
}

#[tokio::test]
async fn retries_when_first_output_is_begin_patch_format() {
    let llm = Arc::new(ScriptedLlm {
        outputs: Mutex::new(vec![
            "*** Begin Patch\n*** Update File: src/lib.rs\n@@\n-pub fn val() -> i32 { 1 }\n+pub fn val() -> i32 { 2 }\n*** End Patch\n"
                .to_string(),
            "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-pub fn val() -> i32 { 1 }\n+pub fn val() -> i32 { 2 }\n"
                .to_string(),
        ]),
    });

    let coder = Coder::new(
        llm,
        CompletionConfig {
            model: "mock".to_string(),
            temperature: 0.0,
            max_tokens: 256,
            json_mode: false,
        },
    );

    let step = PlanStep {
        step: 1,
        task: "Update val function".to_string(),
        target_files: vec!["src/lib.rs".into()],
        depends_on: vec![],
        hypothesis: None,
        predicted_consequences: vec![],
        experiment: None,
        confidence: None,
    };

    let patches = coder
        .generate_patches(
            &step,
            &["pub fn val() -> i32 { 1 }".to_string()],
            &[(
                "src/lib.rs".into(),
                "pub fn val() -> i32 { 1 }\n".to_string(),
            )],
        )
        .await
        .expect("coder should recover from begin patch format");

    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].file, Path::new("src/lib.rs"));
}

#[tokio::test]
async fn retries_when_first_output_missing_hunks() {
    let llm = Arc::new(ScriptedLlm {
        outputs: Mutex::new(vec![
            "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n+pub fn val() -> i32 { 2 }\n"
                .to_string(),
            "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-pub fn val() -> i32 { 1 }\n+pub fn val() -> i32 { 2 }\n"
                .to_string(),
        ]),
    });

    let coder = Coder::new(
        llm,
        CompletionConfig {
            model: "mock".to_string(),
            temperature: 0.0,
            max_tokens: 256,
            json_mode: false,
        },
    );

    let step = PlanStep {
        step: 1,
        task: "Update val function".to_string(),
        target_files: vec!["src/lib.rs".into()],
        depends_on: vec![],
        hypothesis: None,
        predicted_consequences: vec![],
        experiment: None,
        confidence: None,
    };

    let patches = coder
        .generate_patches(
            &step,
            &["pub fn val() -> i32 { 1 }".to_string()],
            &[(
                "src/lib.rs".into(),
                "pub fn val() -> i32 { 1 }\n".to_string(),
            )],
        )
        .await
        .expect("coder should recover by repairing malformed diff");

    assert_eq!(patches.len(), 1);
    assert!(patches[0].diff.contains("@@ -1 +1 @@"));
}

#[tokio::test]
async fn retries_when_first_output_contains_garbage_lines() {
    let llm = Arc::new(ScriptedLlm {
        outputs: Mutex::new(vec![
            "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-pub fn val() -> i32 { 1 }\n+pub fn val() -> i32 { 2 }\nSummary: updated value\n"
                .to_string(),
            "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-pub fn val() -> i32 { 1 }\n+pub fn val() -> i32 { 2 }\n"
                .to_string(),
        ]),
    });

    let coder = Coder::new(
        llm,
        CompletionConfig {
            model: "mock".to_string(),
            temperature: 0.0,
            max_tokens: 256,
            json_mode: false,
        },
    );

    let step = PlanStep {
        step: 1,
        task: "Update val function".to_string(),
        target_files: vec!["src/lib.rs".into()],
        depends_on: vec![],
        hypothesis: None,
        predicted_consequences: vec![],
        experiment: None,
        confidence: None,
    };

    let patches = coder
        .generate_patches(
            &step,
            &["pub fn val() -> i32 { 1 }".to_string()],
            &[(
                "src/lib.rs".into(),
                "pub fn val() -> i32 { 1 }\n".to_string(),
            )],
        )
        .await
        .expect("coder should repair and remove garbage lines from diff");

    assert_eq!(patches.len(), 1);
    assert!(!patches[0].diff.contains("Summary:"));
}

fn run(repo: &Path, shell: &str) {
    let status = Command::new("sh")
        .arg("-lc")
        .arg(shell)
        .current_dir(repo)
        .status()
        .expect("spawn shell command");
    assert!(status.success(), "command failed: {shell}");
}
