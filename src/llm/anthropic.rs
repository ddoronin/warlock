use crate::llm::provider::{CompletionConfig, LlmProvider, Message, Role};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct AnthropicProvider {
	client: Client,
	api_key: String,
	base_url: String,
}

impl AnthropicProvider {
	pub fn new(api_key: impl Into<String>) -> Self {
		Self {
			client: Client::new(),
			api_key: api_key.into(),
			base_url: "https://api.anthropic.com/v1/messages".to_string(),
		}
	}
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
	model: String,
	max_tokens: u32,
	temperature: f32,
	system: String,
	messages: Vec<AnthropicMessage>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
	role: String,
	content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
	content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
	#[serde(rename = "type")]
	kind: String,
	text: Option<String>,
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
	async fn complete(&self, messages: &[Message], config: &CompletionConfig) -> Result<String> {
		let mut system_parts = Vec::new();
		let mut chat_messages = Vec::new();

		for message in messages {
			match message.role {
				Role::System => system_parts.push(message.content.clone()),
				Role::User => chat_messages.push(AnthropicMessage {
					role: "user".to_string(),
					content: message.content.clone(),
				}),
				Role::Assistant => chat_messages.push(AnthropicMessage {
					role: "assistant".to_string(),
					content: message.content.clone(),
				}),
			}
		}

		let req = AnthropicRequest {
			model: config.model.clone(),
			max_tokens: config.max_tokens,
			temperature: config.temperature,
			system: system_parts.join("\n\n"),
			messages: chat_messages,
		};

		let response: AnthropicResponse = self
			.client
			.post(&self.base_url)
			.header("x-api-key", &self.api_key)
			.header("anthropic-version", "2023-06-01")
			.json(&req)
			.send()
			.await
			.context("failed to send Anthropic request")?
			.error_for_status()
			.context("Anthropic API returned error status")?
			.json()
			.await
			.context("failed to parse Anthropic response")?;

		let combined = response
			.content
			.into_iter()
			.filter(|c| c.kind == "text")
			.filter_map(|c| c.text)
			.collect::<Vec<_>>()
			.join("\n");

		Ok(combined)
	}

	fn provider_name(&self) -> &'static str {
		"anthropic"
	}
}
