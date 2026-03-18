use crate::llm::provider::{CompletionConfig, LlmProvider, Message, Role};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info, warn};

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1/chat/completions".to_string(),
        }
    }

    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    content: String,
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn complete(&self, messages: &[Message], config: &CompletionConfig) -> Result<String> {
        let started = Instant::now();
        let use_max_completion_tokens = prefers_max_completion_tokens(&config.model);
        let use_temperature = supports_temperature_param(&config.model);
        if !use_temperature && (config.temperature - 1.0).abs() > f32::EPSILON {
            warn!(
                provider = "openai",
                model = %config.model,
                requested_temperature = config.temperature,
                "model does not support explicit temperature override; using provider default"
            );
        }
        info!(
            provider = "openai",
            model = %config.model,
            json_mode = config.json_mode,
            max_tokens = config.max_tokens,
            message_count = messages.len(),
            "llm request start"
        );
        let request = OpenAiRequest {
            model: config.model.clone(),
            messages: messages
                .iter()
                .map(|m| OpenAiMessage {
                    role: match m.role {
                        Role::System => "system".to_string(),
                        Role::User => "user".to_string(),
                        Role::Assistant => "assistant".to_string(),
                    },
                    content: m.content.clone(),
                })
                .collect(),
            temperature: use_temperature.then_some(config.temperature),
            max_tokens: (!use_max_completion_tokens).then_some(config.max_tokens),
            max_completion_tokens: use_max_completion_tokens.then_some(config.max_tokens),
            // Keep compatibility across OpenAI-compatible backends/models.
            // We still request JSON via prompt discipline and parser recovery.
            response_format: None,
        };

        let request_chars: usize = request.messages.iter().map(|m| m.content.len()).sum();

        info!(
            provider = "openai",
            model = %config.model,
            request_chars,
            request_preview = %preview_openai_request(&request, 2000),
            "llm request payload"
        );

        let http_response = self
            .client
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .context("failed to send OpenAI completion request")?;

        let status = http_response.status();
        let body = http_response
            .text()
            .await
            .context("failed to read OpenAI completion response body")?;
        debug!(
            provider = "openai",
            model = %config.model,
            status = %status,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "llm response received"
        );
        info!(
            provider = "openai",
            model = %config.model,
            response_preview = %truncate_for_error(&body, 2000),
            "llm response body"
        );

        if !status.is_success() {
            warn!(
                provider = "openai",
                model = %config.model,
                status = %status,
                elapsed_ms = started.elapsed().as_millis() as u64,
                body_len = body.len(),
                "llm request failed"
            );
            return Err(anyhow!(
                "OpenAI API returned error status {} for model '{}': {}",
                status,
                config.model,
                truncate_for_error(&body, 1500)
            ));
        }

        let response: OpenAiResponse =
            serde_json::from_str(&body).context("failed to parse OpenAI completion response")?;

        let text = response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();
        info!(
            provider = "openai",
            model = %config.model,
            status = %status,
            elapsed_ms = started.elapsed().as_millis() as u64,
            output_chars = text.len(),
            "llm request succeeded"
        );
        Ok(text)
    }

    fn provider_name(&self) -> &'static str {
        "openai"
    }
}

fn truncate_for_error(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    format!("{}...", &s[..max_len])
}

fn prefers_max_completion_tokens(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    m.starts_with("gpt-5") || m.starts_with("o")
}

fn supports_temperature_param(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    !(m.starts_with("gpt-5") || m.starts_with("o"))
}

fn preview_openai_request(request: &OpenAiRequest, max_len: usize) -> String {
    let json = serde_json::to_string(request).unwrap_or_else(|_| "{}".to_string());
    truncate_for_error(&json, max_len)
}

#[cfg(test)]
mod tests {
    use super::{prefers_max_completion_tokens, supports_temperature_param};

    #[test]
    fn gpt5_prefers_max_completion_tokens() {
        assert!(prefers_max_completion_tokens("gpt-5-mini"));
        assert!(prefers_max_completion_tokens("gpt-5.2"));
    }

    #[test]
    fn gpt5_and_o_models_disable_temperature_param() {
        assert!(!supports_temperature_param("gpt-5-mini"));
        assert!(!supports_temperature_param("o3-mini"));
    }

    #[test]
    fn legacy_models_allow_temperature_param() {
        assert!(supports_temperature_param("gpt-4o-mini"));
        assert!(supports_temperature_param("gpt-4.1"));
    }
}
