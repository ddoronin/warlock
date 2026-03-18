use crate::agents::planner::PlanStep;
use crate::llm::provider::{CompletionConfig, LlmProvider, Message, Role};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodePatch {
    /// Relative file path to create or modify.
    pub file: PathBuf,
    /// Unified diff content.
    pub diff: String,
    /// Whether this patch creates a new file.
    pub is_new_file: bool,
}

pub struct Coder {
    llm: Arc<dyn LlmProvider>,
    completion: CompletionConfig,
}

impl Coder {
    pub fn new(llm: Arc<dyn LlmProvider>, completion: CompletionConfig) -> Self {
        Self { llm, completion }
    }

    pub async fn generate_patches(
        &self,
        step: &PlanStep,
        retrieved_code: &[String],
        target_file_contents: &[(PathBuf, String)],
    ) -> Result<Vec<CodePatch>> {
        let mut cfg = self.completion.clone();
        cfg.json_mode = false;

        let retrieved = retrieved_code.join("\n\n---\n\n");
        let target = target_file_contents
            .iter()
            .map(|(p, c)| format!("FILE: {}\n{}", p.display(), c))
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        let prompt = format!(
			"Task:\n{}\n\nRelevant code:\n{}\n\nCurrent target files:\n{}\n\nReturn only a valid unified diff.",
			step.task, retrieved, target
		);

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are an expert software engineer. Output only unified diff. Do not use the `*** Begin Patch` format."
                    .to_string(),
            },
            Message {
                role: Role::User,
                content: prompt,
            },
        ];

        let raw = self.llm.complete(&messages, &cfg).await?;
        if let Ok(patches) = parse_unified_diff(&raw) {
            if validate_patch_quality(&patches).is_ok() {
                return Ok(patches);
            }
        }

        let repair_prompt = format!(
            "Convert the following assistant output into a valid git unified diff.\n\
             Rules:\n\
             - Return only unified diff text (no markdown fences, no prose).\n\
             - Include at least one file diff.\n\
               - Every file diff must include at least one hunk header (`@@ -old,+new @@`) and changed lines.\n\
             - Preserve the intended changes.\n\
             - If the input is in `*** Begin Patch` format, convert it to unified diff.\n\n\
             Assistant output to convert:\n{raw}\n\n\
             Current target files:\n{target}"
        );

        let repair_messages = vec![
            Message {
                role: Role::System,
                content:
                    "You convert patch-like text into valid git unified diff. Output only diff."
                        .to_string(),
            },
            Message {
                role: Role::User,
                content: repair_prompt,
            },
        ];

        let repaired = self.llm.complete(&repair_messages, &cfg).await?;
        let repaired_patches = parse_unified_diff(&repaired)?;
        validate_patch_quality(&repaired_patches)?;
        Ok(repaired_patches)
    }
}

fn validate_patch_quality(patches: &[CodePatch]) -> Result<()> {
    ensure!(!patches.is_empty(), "no patches generated");

    for patch in patches {
        ensure!(
            patch.diff.contains("--- ") && patch.diff.contains("+++ "),
            "patch for {} is missing file headers",
            patch.file.display()
        );
        ensure!(
            patch.diff.lines().any(is_valid_hunk_header),
            "patch for {} is missing valid hunk headers",
            patch.file.display()
        );
        ensure!(
            !patch
                .diff
                .lines()
                .any(|line| line.starts_with("@@") && !is_valid_hunk_header(line)),
            "patch for {} contains invalid hunk headers",
            patch.file.display()
        );
        ensure!(
            patch.diff.lines().any(|line| {
                line.starts_with('+') || line.starts_with('-') || line.starts_with(' ')
            }),
            "patch for {} has no hunk content",
            patch.file.display()
        );
        ensure!(
            !patch.diff.lines().any(|line| line.trim() == "..."),
            "patch for {} contains placeholder ellipsis",
            patch.file.display()
        );
        ensure!(
            is_syntactically_valid_unified_diff(&patch.diff),
            "patch for {} has invalid unified diff syntax",
            patch.file.display()
        );
    }

    Ok(())
}

fn is_syntactically_valid_unified_diff(diff: &str) -> bool {
    #[derive(Clone, Copy)]
    enum State {
        Header,
        Hunk,
    }

    let mut saw_hunk = false;
    let mut state = State::Header;

    for line in diff.lines() {
        match state {
            State::Header => {
                if is_valid_hunk_header(line) {
                    saw_hunk = true;
                    state = State::Hunk;
                    continue;
                }

                if line.starts_with("diff --git ")
                    || line.starts_with("index ")
                    || line == "new file mode 100644"
                    || line == "deleted file mode 100644"
                    || line.starts_with("--- ")
                    || line.starts_with("+++ ")
                    || line.starts_with("similarity index ")
                    || line.starts_with("rename from ")
                    || line.starts_with("rename to ")
                    || line.starts_with("old mode ")
                    || line.starts_with("new mode ")
                    || line.starts_with("Binary files ")
                    || line.trim().is_empty()
                {
                    continue;
                }

                return false;
            }
            State::Hunk => {
                if is_valid_hunk_header(line) {
                    continue;
                }

                if line.starts_with(' ') || line.starts_with('+') || line.starts_with('-') {
                    continue;
                }

                if line.starts_with("\\ No newline at end of file") {
                    continue;
                }

                if line.starts_with("diff --git ") {
                    state = State::Header;
                    continue;
                }

                return false;
            }
        }
    }

    saw_hunk
}

fn is_valid_hunk_header(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("@@ -") else {
        return false;
    };
    let Some((old_range, after_old)) = rest.split_once(" +") else {
        return false;
    };
    let Some((new_range, _after_new)) = after_old.split_once(" @@") else {
        return false;
    };

    is_valid_hunk_range(old_range) && is_valid_hunk_range(new_range)
}

fn is_valid_hunk_range(range: &str) -> bool {
    if range.is_empty() {
        return false;
    }

    let mut parts = range.split(',');
    let Some(start) = parts.next() else {
        return false;
    };
    if !start.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }

    match parts.next() {
        Some(count) if !count.is_empty() && count.chars().all(|c| c.is_ascii_digit()) => {
            parts.next().is_none()
        }
        None => true,
        _ => false,
    }
}

pub fn parse_unified_diff(raw: &str) -> Result<Vec<CodePatch>> {
    let normalized = normalize_patch_text(raw);
    let lines: Vec<&str> = normalized.lines().collect();
    let mut patches = parse_git_style_blocks(&lines)?;
    if patches.is_empty() {
        patches = parse_plain_unified_blocks(&lines)?;
    }

    ensure!(!patches.is_empty(), "no diff blocks found in coder output");
    Ok(patches)
}

fn normalize_patch_text(raw: &str) -> String {
    if raw.contains("*** Begin Patch") {
        if let Some(converted) = convert_begin_patch_to_unified_diff(raw) {
            return converted;
        }
    }
    raw.to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BeginPatchOp {
    Add,
    Update,
    Delete,
}

fn convert_begin_patch_to_unified_diff(raw: &str) -> Option<String> {
    if !raw.contains("*** Begin Patch") {
        return None;
    }

    let mut converted_blocks = Vec::new();
    let mut current_op: Option<BeginPatchOp> = None;
    let mut current_file: Option<String> = None;
    let mut current_body: Vec<String> = Vec::new();

    let flush_current = |blocks: &mut Vec<String>,
                         op: Option<BeginPatchOp>,
                         file: Option<String>,
                         body: &[String]| {
        let (op, file) = match (op, file) {
            (Some(op), Some(file)) => (op, file),
            _ => return,
        };

        let normalized_body = ensure_hunks_for_begin_patch_body(op, body);

        let mut block = String::new();
        block.push_str(&format!("diff --git a/{file} b/{file}\n"));
        match op {
            BeginPatchOp::Add => {
                block.push_str("new file mode 100644\n");
                block.push_str("--- /dev/null\n");
                block.push_str(&format!("+++ b/{file}\n"));
            }
            BeginPatchOp::Update => {
                block.push_str(&format!("--- a/{file}\n"));
                block.push_str(&format!("+++ b/{file}\n"));
            }
            BeginPatchOp::Delete => {
                block.push_str("deleted file mode 100644\n");
                block.push_str(&format!("--- a/{file}\n"));
                block.push_str("+++ /dev/null\n");
            }
        }

        for line in &normalized_body {
            block.push_str(line);
            block.push('\n');
        }
        blocks.push(block.trim_end_matches('\n').to_string());
    };

    for raw_line in raw.lines() {
        let line = raw_line.trim_end_matches('\r');

        if let Some(file) = line.strip_prefix("*** Add File: ") {
            flush_current(
                &mut converted_blocks,
                current_op,
                current_file.take(),
                &current_body,
            );
            current_body.clear();
            current_op = Some(BeginPatchOp::Add);
            current_file = Some(file.trim().to_string());
            continue;
        }

        if let Some(file) = line.strip_prefix("*** Update File: ") {
            flush_current(
                &mut converted_blocks,
                current_op,
                current_file.take(),
                &current_body,
            );
            current_body.clear();
            current_op = Some(BeginPatchOp::Update);
            current_file = Some(file.trim().to_string());
            continue;
        }

        if let Some(file) = line.strip_prefix("*** Delete File: ") {
            flush_current(
                &mut converted_blocks,
                current_op,
                current_file.take(),
                &current_body,
            );
            current_body.clear();
            current_op = Some(BeginPatchOp::Delete);
            current_file = Some(file.trim().to_string());
            continue;
        }

        if line == "*** Begin Patch" || line == "*** End Patch" {
            continue;
        }

        if line.starts_with("*** ") {
            // Ignore unrecognized patch directives like section hints.
            continue;
        }

        if current_file.is_some() {
            current_body.push(line.to_string());
        }
    }

    flush_current(
        &mut converted_blocks,
        current_op,
        current_file.take(),
        &current_body,
    );

    if converted_blocks.is_empty() {
        None
    } else {
        Some(converted_blocks.join("\n"))
    }
}

fn ensure_hunks_for_begin_patch_body(op: BeginPatchOp, body: &[String]) -> Vec<String> {
    let has_valid_headers = body.iter().any(|line| is_valid_hunk_header(line));

    let normalize_line = |line: &str| -> Option<String> {
        if line.starts_with("@@") {
            if is_valid_hunk_header(line) {
                return Some(line.to_string());
            }
            // Section hints like "@@ fn foo" are not valid unified headers.
            return None;
        }

        if line.starts_with('+') || line.starts_with('-') || line.starts_with(' ') {
            return Some(line.to_string());
        }

        let prefixed = match op {
            BeginPatchOp::Add => format!("+{line}"),
            BeginPatchOp::Delete => format!("-{line}"),
            BeginPatchOp::Update => format!(" {line}"),
        };
        Some(prefixed)
    };

    let mut normalized = Vec::new();
    for line in body {
        if let Some(line) = normalize_line(line) {
            normalized.push(line);
        }
    }

    if has_valid_headers {
        return normalized;
    }

    let has_changes = normalized
        .iter()
        .any(|line| line.starts_with('+') || line.starts_with('-'));
    if !has_changes {
        return normalized;
    }

    let old_count = normalized
        .iter()
        .filter(|line| !line.starts_with('+'))
        .count();
    let new_count = normalized
        .iter()
        .filter(|line| !line.starts_with('-'))
        .count();

    let old_start = if old_count == 0 { 0 } else { 1 };
    let new_start = if new_count == 0 { 0 } else { 1 };

    let mut with_hunk = Vec::with_capacity(normalized.len() + 1);
    with_hunk.push(format!(
        "@@ -{},{} +{},{} @@",
        old_start, old_count, new_start, new_count
    ));
    with_hunk.extend(normalized);
    with_hunk
}

fn parse_git_style_blocks(lines: &[&str]) -> Result<Vec<CodePatch>> {
    let mut i = 0usize;
    let mut patches = Vec::new();

    while i < lines.len() {
        if !lines[i].starts_with("diff --git ") {
            i += 1;
            continue;
        }

        let start = i;
        i += 1;
        while i < lines.len() && !lines[i].starts_with("diff --git ") {
            i += 1;
        }

        let block = lines[start..i].join("\n");
        patches.push(parse_diff_block(&block)?);
    }

    Ok(patches)
}

fn parse_plain_unified_blocks(lines: &[&str]) -> Result<Vec<CodePatch>> {
    let mut i = 0usize;
    let mut patches = Vec::new();

    while i < lines.len() {
        if !lines[i].starts_with("--- ") {
            i += 1;
            continue;
        }

        let start = i;
        i += 1;
        if i >= lines.len() || !lines[i].starts_with("+++ ") {
            continue;
        }

        i += 1;
        while i < lines.len()
            && !lines[i].starts_with("--- ")
            && !lines[i].starts_with("diff --git ")
        {
            i += 1;
        }

        let block = lines[start..i].join("\n");
        patches.push(parse_diff_block(&block)?);
    }

    Ok(patches)
}

fn parse_diff_block(block: &str) -> Result<CodePatch> {
    let mut file: Option<PathBuf> = None;
    let mut is_new_file = false;

    for line in block.lines() {
        if line.starts_with("+++ ") {
            let new_path = line.trim_start_matches("+++ ").trim();
            if new_path == "/dev/null" {
                continue;
            }
            let normalized = new_path
                .strip_prefix("b/")
                .unwrap_or(new_path)
                .trim()
                .to_string();
            file = Some(PathBuf::from(normalized));
        } else if line.starts_with("--- ") {
            let old_path = line.trim_start_matches("--- ").trim();
            if old_path == "/dev/null" {
                is_new_file = true;
            }
        } else if line == "new file mode 100644" {
            is_new_file = true;
        }
    }

    let file = file.ok_or_else(|| anyhow::anyhow!("diff block missing +++ path"))?;
    let diff = if block.ends_with('\n') {
        block.to_string()
    } else {
        format!("{block}\n")
    };

    Ok(CodePatch {
        file,
        diff,
        is_new_file,
    })
}
