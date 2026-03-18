use crate::config::IndexingConfig;
use anyhow::{Context, Result};
use ignore::gitignore::GitignoreBuilder;
use ignore::WalkBuilder;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const BINARY_EXTENSIONS: &[&str] = &[
	"png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "pdf", "zip", "gz", "xz", "7z", "tar",
	"jar", "class", "so", "dylib", "dll", "a", "o", "exe", "bin",
];

pub fn discover_files(repo_root: &Path, config: &IndexingConfig) -> Result<Vec<PathBuf>> {
	let mut ignore_builder = GitignoreBuilder::new(repo_root);
	for pattern in &config.ignore_patterns {
		ignore_builder
			.add_line(None, pattern)
			.with_context(|| format!("invalid ignore pattern: {pattern}"))?;
	}
	let custom_ignore = ignore_builder.build().context("failed to build ignore matcher")?;

	let mut builder = WalkBuilder::new(repo_root);
	builder
		.git_ignore(true)
		.git_global(true)
		.git_exclude(true)
		.hidden(false)
		.follow_links(false)
		.parents(true);

	let mut files = Vec::new();
	for entry in builder.build() {
		let entry = match entry {
			Ok(entry) => entry,
			Err(_) => continue,
		};

		let path = entry.path();
		if !entry
			.file_type()
			.map(|ft| ft.is_file())
			.unwrap_or_else(|| path.is_file())
		{
			continue;
		}

		let rel = match path.strip_prefix(repo_root) {
			Ok(p) => p,
			Err(_) => continue,
		};

		if custom_ignore.matched(rel, false).is_ignore() {
			continue;
		}

		if is_obvious_binary(path) {
			continue;
		}

		let max_size = config.max_file_size_kb.saturating_mul(1024);
		let metadata = match path.metadata() {
			Ok(meta) => meta,
			Err(_) => continue,
		};
		if metadata.len() > max_size {
			continue;
		}

		files.push(path.to_path_buf());
	}

	files.sort();
	Ok(files)
}

fn is_obvious_binary(path: &Path) -> bool {
	let extension = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
	if let Some(ext) = extension.as_deref() {
		if BINARY_EXTENSIONS.contains(&ext) {
			return true;
		}
	}

	let mut file = match File::open(path) {
		Ok(file) => file,
		Err(_) => return true,
	};

	let mut buf = [0u8; 1024];
	let n = match file.read(&mut buf) {
		Ok(n) => n,
		Err(_) => return true,
	};

	buf[..n].contains(&0)
}
