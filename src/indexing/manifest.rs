use crate::config::IndexingConfig;
use crate::indexing::walker;
use crate::retrieval::vector_store::{normalize_repo_url, repository_remote_url};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestFileEntry {
    pub path: String,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepositoryManifest {
    pub repo_id: String,
    #[serde(default = "default_identity_version")]
    pub identity_version: u32,
    #[serde(default)]
    pub repo_url: Option<String>,
    pub commit_hash: Option<String>,
    pub indexed_at_unix: u64,
    pub embedding_model: String,
    pub embedding_dimensions: u32,
    pub files: Vec<ManifestFileEntry>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ManifestDiff {
    pub changed_or_new: Vec<String>,
    pub removed: Vec<String>,
    pub requires_full_reindex: bool,
}

impl RepositoryManifest {
    pub fn default_path(repo_root: &Path) -> PathBuf {
        repo_root.join(".warlock/repo_index.json")
    }

    pub fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }

        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read manifest {}", path.display()))?;
        let manifest = serde_json::from_str::<Self>(&raw)
            .with_context(|| format!("failed to parse manifest {}", path.display()))?;
        Ok(Some(manifest))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create manifest dir {}", parent.display()))?;
        }

        let raw = serde_json::to_string_pretty(self).context("failed to serialize manifest")?;
        fs::write(path, raw).with_context(|| format!("failed to write manifest {}", path.display()))
    }

    pub fn build(
        repo_root: &Path,
        config: &IndexingConfig,
        repo_id: &str,
        embedding_model: &str,
        embedding_dimensions: u32,
    ) -> Result<Self> {
        let files = walker::discover_files(repo_root, config)?;
        let mut file_entries = Vec::with_capacity(files.len());

        for file in files {
            let rel = file
                .strip_prefix(repo_root)
                .with_context(|| format!("failed to strip prefix for {}", file.display()))?;
            let bytes = fs::read(&file)
                .with_context(|| format!("failed to read file for hashing: {}", file.display()))?;
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            file_entries.push(ManifestFileEntry {
                path: rel.to_string_lossy().to_string(),
                hash: hex::encode(hasher.finalize()),
            });
        }

        file_entries.sort_by(|a, b| a.path.cmp(&b.path));

        Ok(Self {
            repo_id: repo_id.to_string(),
            identity_version: default_identity_version(),
            repo_url: repository_remote_url(repo_root).map(|u| normalize_repo_url(&u)),
            commit_hash: git_commit_hash(repo_root),
            indexed_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            embedding_model: embedding_model.to_string(),
            embedding_dimensions,
            files: file_entries,
        })
    }

    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.repo_id == other.repo_id
            && self.identity_version == other.identity_version
            && self.repo_url == other.repo_url
            && self.embedding_model == other.embedding_model
            && self.embedding_dimensions == other.embedding_dimensions
            && self.files == other.files
    }

    pub fn diff_from(&self, previous: Option<&Self>) -> ManifestDiff {
        let mut diff = ManifestDiff::default();

        let Some(previous) = previous else {
            diff.changed_or_new = self.files.iter().map(|f| f.path.clone()).collect();
            return diff;
        };

        if previous.repo_id != self.repo_id
            || previous.identity_version != self.identity_version
            || previous.repo_url != self.repo_url
            || previous.embedding_model != self.embedding_model
            || previous.embedding_dimensions != self.embedding_dimensions
        {
            diff.requires_full_reindex = true;
            diff.changed_or_new = self.files.iter().map(|f| f.path.clone()).collect();
            return diff;
        }

        let prev_map: HashMap<&str, &str> = previous
            .files
            .iter()
            .map(|f| (f.path.as_str(), f.hash.as_str()))
            .collect();
        let curr_map: HashMap<&str, &str> = self
            .files
            .iter()
            .map(|f| (f.path.as_str(), f.hash.as_str()))
            .collect();

        for f in &self.files {
            match prev_map.get(f.path.as_str()) {
                Some(prev_hash) if *prev_hash == f.hash.as_str() => {}
                _ => diff.changed_or_new.push(f.path.clone()),
            }
        }

        for f in &previous.files {
            if !curr_map.contains_key(f.path.as_str()) {
                diff.removed.push(f.path.clone());
            }
        }

        diff.changed_or_new.sort();
        diff.changed_or_new.dedup();
        diff.removed.sort();
        diff.removed.dedup();

        diff
    }
}

fn default_identity_version() -> u32 {
    1
}

fn git_commit_hash(repo_root: &Path) -> Option<String> {
    let repo = git2::Repository::open(repo_root).ok()?;
    let head = repo.head().ok()?;
    let oid = head.target()?;
    Some(oid.to_string())
}

#[cfg(test)]
mod tests {
    use super::{ManifestDiff, ManifestFileEntry, RepositoryManifest};

    #[test]
    fn compatibility_checks_files_and_embedding_settings() {
        let left = RepositoryManifest {
            repo_id: "repo_x".to_string(),
            identity_version: 1,
            repo_url: Some("github.com/acme/warlock".to_string()),
            commit_hash: Some("abc".to_string()),
            indexed_at_unix: 1,
            embedding_model: "text-embedding-3-small".to_string(),
            embedding_dimensions: 1536,
            files: vec![ManifestFileEntry {
                path: "src/main.rs".to_string(),
                hash: "h1".to_string(),
            }],
        };

        let mut right = left.clone();
        right.commit_hash = Some("def".to_string());
        right.indexed_at_unix = 2;
        assert!(left.is_compatible_with(&right));

        right.files[0].hash = "h2".to_string();
        assert!(!left.is_compatible_with(&right));
    }

    #[test]
    fn diff_detects_changes_and_deletions() {
        let previous = RepositoryManifest {
            repo_id: "repo_x".to_string(),
            identity_version: 1,
            repo_url: Some("github.com/acme/warlock".to_string()),
            commit_hash: Some("abc".to_string()),
            indexed_at_unix: 1,
            embedding_model: "text-embedding-3-small".to_string(),
            embedding_dimensions: 1536,
            files: vec![
                ManifestFileEntry {
                    path: "src/a.rs".to_string(),
                    hash: "h1".to_string(),
                },
                ManifestFileEntry {
                    path: "src/b.rs".to_string(),
                    hash: "h2".to_string(),
                },
            ],
        };

        let current = RepositoryManifest {
            repo_id: "repo_x".to_string(),
            identity_version: 1,
            repo_url: Some("github.com/acme/warlock".to_string()),
            commit_hash: Some("def".to_string()),
            indexed_at_unix: 2,
            embedding_model: "text-embedding-3-small".to_string(),
            embedding_dimensions: 1536,
            files: vec![
                ManifestFileEntry {
                    path: "src/a.rs".to_string(),
                    hash: "h1_changed".to_string(),
                },
                ManifestFileEntry {
                    path: "src/c.rs".to_string(),
                    hash: "h3".to_string(),
                },
            ],
        };

        let diff = current.diff_from(Some(&previous));
        assert_eq!(
            diff,
            ManifestDiff {
                changed_or_new: vec!["src/a.rs".to_string(), "src/c.rs".to_string()],
                removed: vec!["src/b.rs".to_string()],
                requires_full_reindex: false,
            }
        );
    }
}
