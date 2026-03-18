use crate::llm::provider::{CompletionConfig, LlmProvider, Message, Role};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OllamaProvider {
	client: Client,
	base_url: String,
}

impl OllamaProvider {
	pub fn new(base_url: impl Into<String>) -> Self {
		Self {
			client: Client::new(),
			base_url: base_url.into(),
		}
	}
}

#[derive(Debug, Serialize)]
struct OllamaRequest {
	model: String,
	messages: Vec<OllamaMessage>,
	stream: bool,
	format: Option<String>,
	options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
	role: String,
	content: String,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
	temperature: f32,
	num_predict: u32,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
	message: OllamaResponseMessage,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseMessage {
	content: String,
}

#[async_trait]
impl LlmProvider for OllamaProvider {
	async fn complete(&self, messages: &[Message], config: &CompletionConfig) -> Result<String> {
		let req = OllamaRequest {
			model: config.model.clone(),
			messages: messages
				.iter()
				.map(|m| OllamaMessage {
					role: match m.role {
						Role::System => "system".to_string(),
						Role::User => "user".to_string(),
						Role::Assistant => "assistant".to_string(),
					},
					content: m.content.clone(),
				})
				.collect(),
			stream: false,
			format: config.json_mode.then(|| "json".to_string()),
			options: OllamaOptions {
				temperature: config.temperature,
				num_predict: config.max_tokens,
			},
		};

		let endpoint = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
		let response: OllamaResponse = self
			.client
			.post(endpoint)
			.json(&req)
			.send()
			.await
			.context("failed to send Ollama request")?
			.error_for_status()
			.context("Ollama API returned error status")?
			.json()
			.await
			.context("failed to parse Ollama response")?;

		Ok(response.message.content)
	}

	fn provider_name(&self) -> &'static str {
		"ollama"
	}
}
