use crate::agents::coder::{parse_unified_diff, CodePatch};
use crate::llm::provider::{CompletionConfig, LlmProvider, Message, Role};
use anyhow::Result;
use std::sync::Arc;

pub enum ReflectionOutcome {
	Corrected(Vec<CodePatch>),
	Escalate(String),
}

pub struct Reflector {
	llm: Arc<dyn LlmProvider>,
	completion: CompletionConfig,
}

impl Reflector {
	pub fn new(llm: Arc<dyn LlmProvider>, completion: CompletionConfig) -> Self {
		Self { llm, completion }
	}

	pub async fn reflect(
		&self,
		task: &str,
		failed_diff: &str,
		error_output: &str,
		attempt_history: &str,
	) -> Result<ReflectionOutcome> {
		let mut cfg = self.completion.clone();
		cfg.json_mode = false;

		let messages = vec![
			Message {
				role: Role::System,
				content: "You are a strict code reviewer. Return corrected unified diff or ESCALATE.".to_string(),
			},
			Message {
				role: Role::User,
				content: format!(
					"Task:\n{task}\n\nFailed patch:\n{failed_diff}\n\nErrors:\n{error_output}\n\nAttempt history:\n{attempt_history}\n\nReturn corrected unified diff, or start response with ESCALATE if unresolvable."
				),
			},
		];

		let raw = self.llm.complete(&messages, &cfg).await?;
		let trimmed = raw.trim();
		if trimmed.starts_with("ESCALATE") {
			return Ok(ReflectionOutcome::Escalate(trimmed.to_string()));
		}

		let patches = parse_unified_diff(trimmed)?;
		Ok(ReflectionOutcome::Corrected(patches))
	}
}
