use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub embeddings: EmbeddingsConfig,
    pub vector_store: VectorStoreConfig,
    pub sandbox: SandboxConfig,
    pub agent: AgentConfig,
    pub indexing: IndexingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsConfig {
    pub provider: String,
    pub model: String,
    pub dimensions: u32,
    pub batch_size: usize,
    pub max_concurrency: usize,
    #[serde(default = "default_true")]
    pub persist_cache: bool,
    #[serde(default)]
    pub cache_db_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    pub provider: String,
    pub url: String,
    pub collection: String,
    pub top_k: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    #[serde(default)]
    pub backend: SandboxBackend,
    pub provider: String,
    pub image: String,
    pub timeout_secs: u64,
    pub cpu_limit: u32,
    pub memory_limit_mb: u64,
    pub network_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SandboxBackend {
    Local,
    Docker,
}

impl Default for SandboxBackend {
    fn default() -> Self {
        Self::Local
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub max_reflection_attempts: u32,
    pub planner_max_steps: u32,
    pub coder_context_chunks: u32,
    #[serde(default = "default_agent_plan_cycles")]
    pub max_plan_cycles: u32,
    #[serde(default = "default_true")]
    pub planning_experiments_enabled: bool,
    #[serde(default = "default_true")]
    pub planning_experiment_strict: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    pub max_file_size_kb: u64,
    pub max_chunk_lines: usize,
    #[serde(default = "default_parallel_file_workers")]
    pub parallel_file_workers: usize,
    pub ignore_patterns: Vec<String>,
    pub supported_languages: Vec<String>,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let raw = std::fs::read_to_string(path_ref)
            .with_context(|| format!("failed to read config file: {}", path_ref.display()))?;
        let mut config: Self = toml::from_str(&raw)
            .with_context(|| format!("failed to parse TOML config: {}", path_ref.display()))?;
        config.apply_env_overrides()?;
        config.validate()?;
        Ok(config)
    }

    fn apply_env_overrides(&mut self) -> Result<()> {
        if let Some(url) = std::env::var("WARLOCK_QDRANT_URL")
            .ok()
            .or_else(|| std::env::var("QDRANT_URL").ok())
        {
            if !url.trim().is_empty() {
                self.vector_store.url = url;
            }
        }

        if let Ok(collection) = std::env::var("WARLOCK_QDRANT_COLLECTION") {
            if !collection.trim().is_empty() {
                self.vector_store.collection = collection;
            }
        }

        if let Ok(top_k) = std::env::var("WARLOCK_QDRANT_TOP_K") {
            let top_k = top_k
                .parse::<u64>()
                .with_context(|| "WARLOCK_QDRANT_TOP_K must be a positive integer")?;
            self.vector_store.top_k = top_k;
        }

        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.llm.provider.trim().is_empty(),
            "llm.provider must not be empty"
        );
        ensure!(
            !self.llm.model.trim().is_empty(),
            "llm.model must not be empty"
        );
        ensure!(self.llm.max_tokens > 0, "llm.max_tokens must be > 0");

        ensure!(
            !self.embeddings.provider.trim().is_empty(),
            "embeddings.provider must not be empty"
        );
        ensure!(
            !self.embeddings.model.trim().is_empty(),
            "embeddings.model must not be empty"
        );
        ensure!(
            self.embeddings.dimensions > 0,
            "embeddings.dimensions must be > 0"
        );
        ensure!(
            self.embeddings.batch_size > 0,
            "embeddings.batch_size must be > 0"
        );
        ensure!(
            self.embeddings.max_concurrency > 0,
            "embeddings.max_concurrency must be > 0"
        );
        if let Some(path) = &self.embeddings.cache_db_path {
            ensure!(
                !path.trim().is_empty(),
                "embeddings.cache_db_path must not be empty when set"
            );
        }

        ensure!(
            !self.vector_store.provider.trim().is_empty(),
            "vector_store.provider must not be empty"
        );
        ensure!(
            !self.vector_store.url.trim().is_empty(),
            "vector_store.url must not be empty"
        );
        ensure!(
            !self.vector_store.collection.trim().is_empty(),
            "vector_store.collection must not be empty"
        );
        ensure!(
            self.vector_store.top_k > 0,
            "vector_store.top_k must be > 0"
        );

        ensure!(
            self.sandbox.timeout_secs > 0,
            "sandbox.timeout_secs must be > 0"
        );
        match self.sandbox.backend {
            SandboxBackend::Local => {}
            SandboxBackend::Docker => {
                ensure!(
                    !self.sandbox.provider.trim().is_empty(),
                    "sandbox.provider must not be empty when sandbox.backend=docker"
                );
                ensure!(
                    self.sandbox.provider.eq_ignore_ascii_case("docker"),
                    "sandbox.provider must be 'docker' when sandbox.backend=docker"
                );
                ensure!(
                    !self.sandbox.image.trim().is_empty(),
                    "sandbox.image must not be empty when sandbox.backend=docker"
                );
                ensure!(
                    self.sandbox.cpu_limit > 0,
                    "sandbox.cpu_limit must be > 0 when sandbox.backend=docker"
                );
                ensure!(
                    self.sandbox.memory_limit_mb > 0,
                    "sandbox.memory_limit_mb must be > 0 when sandbox.backend=docker"
                );
            }
        }

        ensure!(
            self.agent.max_reflection_attempts > 0,
            "agent.max_reflection_attempts must be > 0"
        );
        ensure!(
            self.agent.planner_max_steps > 0,
            "agent.planner_max_steps must be > 0"
        );
        ensure!(
            self.agent.coder_context_chunks > 0,
            "agent.coder_context_chunks must be > 0"
        );
        ensure!(
            self.agent.max_plan_cycles > 0,
            "agent.max_plan_cycles must be > 0"
        );

        ensure!(
            self.indexing.max_file_size_kb > 0,
            "indexing.max_file_size_kb must be > 0"
        );
        ensure!(
            self.indexing.max_chunk_lines > 0,
            "indexing.max_chunk_lines must be > 0"
        );
        ensure!(
            self.indexing.parallel_file_workers > 0,
            "indexing.parallel_file_workers must be > 0"
        );
        ensure!(
            !self.indexing.supported_languages.is_empty(),
            "indexing.supported_languages must not be empty"
        );

        Ok(())
    }
}

fn default_true() -> bool {
    true
}

fn default_agent_plan_cycles() -> u32 {
    2
}

fn default_parallel_file_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::Config;
    use tempfile::NamedTempFile;

    #[test]
    fn loads_valid_warlock_toml() {
        let mut file = NamedTempFile::new().expect("temp file");
        std::io::Write::write_all(&mut file, include_str!("../warlock.toml").as_bytes())
            .expect("write config");

        let cfg = Config::load(file.path()).expect("config should load");
        assert_eq!(cfg.llm.provider, "openai");
        assert_eq!(cfg.embeddings.dimensions, 1536);
        assert!(!cfg.indexing.supported_languages.is_empty());
    }
}
