use crate::indexing::CodeChunk;
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

#[derive(Clone, Default)]
pub struct EmbeddingCache {
    inner: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    db: Option<Arc<sled::Db>>,
}

impl EmbeddingCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_db(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create cache db directory {}", parent.display())
            })?;
        }

        let db = sled::open(path_ref)
            .with_context(|| format!("failed to open embedding cache db {}", path_ref.display()))?;

        Ok(Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            db: Some(Arc::new(db)),
        })
    }

    pub fn key_for_text(text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hex::encode(hasher.finalize())
    }

    pub fn key_for_chunk(chunk: &CodeChunk) -> String {
        let mut hasher = Sha256::new();
        hasher.update(chunk.file.to_string_lossy().as_bytes());
        hasher.update(b"\x1f");
        hasher.update(chunk.symbol.as_bytes());
        hasher.update(b"\x1f");
        hasher.update(chunk.code.as_bytes());
        hex::encode(hasher.finalize())
    }

    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        if let Some(embedding) = self.inner.read().ok().and_then(|m| m.get(key).cloned()) {
            return Some(embedding);
        }

        let db = self.db.as_ref()?;
        let bytes = db.get(key.as_bytes()).ok().flatten()?;
        let parsed = serde_json::from_slice::<Vec<f32>>(&bytes).ok()?;

        if let Ok(mut m) = self.inner.write() {
            m.insert(key.to_string(), parsed.clone());
        }

        Some(parsed)
    }

    pub fn insert(&self, key: impl Into<String>, embedding: Vec<f32>) {
        let key = key.into();

        if let Ok(mut m) = self.inner.write() {
            m.insert(key.clone(), embedding.clone());
        }

        if let Some(db) = &self.db {
            if let Ok(encoded) = serde_json::to_vec(&embedding) {
                let _ = db.insert(key.as_bytes(), encoded);
                let _ = db.flush_async();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EmbeddingCache;
    use tempfile::tempdir;

    #[test]
    fn persists_embeddings_between_instances() {
        let dir = tempdir().expect("temp dir");
        let db_path = dir.path().join("embeddings_cache");

        {
            let cache = EmbeddingCache::with_db(&db_path).expect("open cache");
            cache.insert("k1", vec![0.1, 0.2, 0.3]);
        }

        let cache = EmbeddingCache::with_db(&db_path).expect("reopen cache");
        let value = cache.get("k1").expect("value should be present");
        assert_eq!(value, vec![0.1, 0.2, 0.3]);
    }
}
