use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
	System,
	User,
	Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
	pub role: Role,
	pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionConfig {
	pub model: String,
	pub temperature: f32,
	pub max_tokens: u32,
	pub json_mode: bool,
}

impl CompletionConfig {
	pub fn low_temp_json(model: impl Into<String>, max_tokens: u32) -> Self {
		Self {
			model: model.into(),
			temperature: 0.1,
			max_tokens,
			json_mode: true,
		}
	}
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
	async fn complete(&self, messages: &[Message], config: &CompletionConfig) -> Result<String>;
	fn provider_name(&self) -> &'static str;
}
