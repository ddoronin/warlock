use anyhow::{Context, Result};
use git2::{build::CheckoutBuilder, Repository, ResetType};
use std::path::Path;
use std::process::Command;

pub fn revert_applied_patches(repo_path: &Path) -> Result<()> {
	let repo = Repository::open(repo_path)
		.with_context(|| format!("failed to open git repository: {}", repo_path.display()))?;

	let head = repo
		.head()
		.context("failed to read HEAD")?
		.peel_to_commit()
		.context("failed to peel HEAD to commit")?;

	repo.reset(head.as_object(), ResetType::Hard, None)
		.context("failed to hard reset repository")?;

	repo.checkout_head(Some(
		CheckoutBuilder::new()
			.force()
			.remove_untracked(true)
			.remove_ignored(false),
	))
	.context("failed to checkout HEAD with cleanup")?;

	let _ = Command::new("git")
		.arg("clean")
		.arg("-fd")
		.current_dir(repo_path)
		.output();

	Ok(())
}
