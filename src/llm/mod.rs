pub mod anthropic;
pub mod ollama;
pub mod openai;
pub mod provider;

use crate::llm::anthropic::AnthropicProvider;
use crate::llm::ollama::OllamaProvider;
use crate::llm::openai::OpenAiProvider;
use crate::llm::provider::LlmProvider;
use anyhow::{anyhow, Result};
use std::sync::Arc;

pub fn build_provider(provider_name: &str) -> Result<Arc<dyn LlmProvider>> {
	match provider_name.trim().to_ascii_lowercase().as_str() {
		"openai" => {
			let api_key = std::env::var("OPENAI_API_KEY")
				.map_err(|_| anyhow!("OPENAI_API_KEY is required for openai provider"))?;
			Ok(Arc::new(OpenAiProvider::new(api_key)))
		}
		"anthropic" => {
			let api_key = std::env::var("ANTHROPIC_API_KEY")
				.map_err(|_| anyhow!("ANTHROPIC_API_KEY is required for anthropic provider"))?;
			Ok(Arc::new(AnthropicProvider::new(api_key)))
		}
		"ollama" => {
			let base_url =
				std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
			Ok(Arc::new(OllamaProvider::new(base_url)))
		}
		other => Err(anyhow!("unsupported llm provider: {other}")),
	}
}
