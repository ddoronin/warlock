use anyhow::{bail, Context, Result};
use git2::{ApplyLocation, Diff, Repository};
use std::path::Path;
use std::process::Command;

pub fn apply_patch(repo_path: &Path, diff: &str) -> Result<()> {
	if try_apply_with_git2(repo_path, diff).is_ok() {
		return Ok(());
	}

	apply_patch_with_git_cli(repo_path, diff)
}

fn try_apply_with_git2(repo_path: &Path, diff: &str) -> Result<()> {
	let repo = Repository::open(repo_path)
		.with_context(|| format!("failed to open git repository: {}", repo_path.display()))?;
	let parsed_diff = Diff::from_buffer(diff.as_bytes()).context("invalid diff for git2 apply")?;
	repo.apply(&parsed_diff, ApplyLocation::WorkDir, None)
		.context("git2 apply failed")?;
	Ok(())
}

fn apply_patch_with_git_cli(repo_path: &Path, diff: &str) -> Result<()> {
	let check = Command::new("git")
		.arg("apply")
		.arg("--check")
		.arg("-")
		.current_dir(repo_path)
		.stdin(std::process::Stdio::piped())
		.stdout(std::process::Stdio::piped())
		.stderr(std::process::Stdio::piped())
		.spawn()
		.context("failed to spawn `git apply --check`")?;
	let check_output = write_stdin_and_wait(check, diff)?;
	if !check_output.status.success() {
		bail!(
			"`git apply --check` failed: {}",
			String::from_utf8_lossy(&check_output.stderr)
		);
	}

	let apply = Command::new("git")
		.arg("apply")
		.arg("-")
		.current_dir(repo_path)
		.stdin(std::process::Stdio::piped())
		.stdout(std::process::Stdio::piped())
		.stderr(std::process::Stdio::piped())
		.spawn()
		.context("failed to spawn `git apply`")?;
	let apply_output = write_stdin_and_wait(apply, diff)?;
	if !apply_output.status.success() {
		bail!(
			"`git apply` failed: {}",
			String::from_utf8_lossy(&apply_output.stderr)
		);
	}

	Ok(())
}

fn write_stdin_and_wait(mut child: std::process::Child, input: &str) -> Result<std::process::Output> {
	use std::io::Write;

	if let Some(mut stdin) = child.stdin.take() {
		stdin
			.write_all(input.as_bytes())
			.context("failed writing patch to git stdin")?;
	}
	let output = child
		.wait_with_output()
		.context("failed waiting for git apply process")?;
	Ok(output)
}
