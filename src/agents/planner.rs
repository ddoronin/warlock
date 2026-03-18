use crate::llm::provider::{CompletionConfig, LlmProvider, Message, Role};
use anyhow::{ensure, Context, Result};
use serde::{de::Deserializer, Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentSpec {
    /// Lightweight pre-implementation command to validate the hypothesis.
    pub command: String,
    /// Optional output substrings that must appear for experiment to be considered successful.
    #[serde(default)]
    pub success_contains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanStep {
    /// 1-based step index.
    pub step: u32,
    /// Atomic work item description.
    pub task: String,
    /// Expected files to touch in this step.
    pub target_files: Vec<PathBuf>,
    /// Step dependencies by `step` number.
    pub depends_on: Vec<u32>,
    /// Optional explicit hypothesis for the change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis: Option<String>,
    /// Optional predicted consequences of applying this step.
    #[serde(default)]
    pub predicted_consequences: Vec<String>,
    /// Optional pre-implementation experiment to validate hypothesis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment: Option<ExperimentSpec>,
    /// Optional confidence score in range [0.0, 1.0].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Plan {
    pub goal: String,
    pub steps: Vec<PlanStep>,
}

pub struct Planner {
    llm: Arc<dyn LlmProvider>,
    completion: CompletionConfig,
}

impl Planner {
    pub fn new(llm: Arc<dyn LlmProvider>, completion: CompletionConfig) -> Self {
        Self { llm, completion }
    }

    pub async fn generate_plan(&self, goal: &str, codebase_summary: &str) -> Result<Plan> {
        self.generate_plan_with_limit(goal, codebase_summary, None)
            .await
    }

    pub async fn generate_plan_with_limit(
        &self,
        goal: &str,
        codebase_summary: &str,
        max_steps: Option<u32>,
    ) -> Result<Plan> {
        let messages = vec![
            Message {
                role: Role::System,
                content: "You are a software architect. Return only JSON.".to_string(),
            },
            Message {
                role: Role::User,
                content: format!(
                    "Goal:\n{goal}\n\nCodebase summary:\n{codebase_summary}\n\nReturn JSON array with objects: step, task, target_files, depends_on. Optional fields for each step: hypothesis, predicted_consequences (array of strings), experiment (object with command and success_contains array), confidence (0.0-1.0)."
                ),
            },
        ];

        let mut config = self.completion.clone();
        config.json_mode = true;
        info!(
            model = %config.model,
            goal_len = goal.len(),
            summary_len = codebase_summary.len(),
            max_steps = max_steps.unwrap_or(0),
            "planner llm call start"
        );
        let raw = self.llm.complete(&messages, &config).await?;
        debug!(raw_len = raw.len(), "planner llm response received");

        let mut candidate_raw = raw;
        for attempt in 0..=2 {
            let mut plan = match parse_plan_response(goal, &candidate_raw) {
                Ok(plan) => plan,
                Err(parse_err) => {
                    if attempt < 2 {
                        let parse_error = parse_err.to_string();
                        warn!(
                            error = %parse_error,
                            attempt,
                            "planner parse failed; attempting repair retry"
                        );
                        candidate_raw = self
                            .repair_plan_response(
                                goal,
                                codebase_summary,
                                &candidate_raw,
                                &parse_error,
                                &config,
                                attempt >= 1,
                            )
                            .await?;
                        debug!(
                            raw_len = candidate_raw.len(),
                            "planner repair llm response received"
                        );
                        continue;
                    }
                    return Err(parse_err);
                }
            };

            normalize_step_numbers(&mut plan);
            if let Some(limit) = max_steps {
                enforce_plan_limits(&mut plan, limit);
            }
            if let Err(validation_err) = validate_plan(&plan) {
                if attempt < 2 {
                    let validation_error = validation_err.to_string();
                    warn!(
                        error = %validation_error,
                        attempt,
                        "planner validation failed; attempting repair retry"
                    );
                    candidate_raw = self
                        .repair_plan_response(
                            goal,
                            codebase_summary,
                            &candidate_raw,
                            &validation_error,
                            &config,
                            attempt >= 1,
                        )
                        .await?;
                    debug!(
                        raw_len = candidate_raw.len(),
                        "planner repair llm response received"
                    );
                    continue;
                }
                warn!(error = %validation_err, "planner validation failed after repair");
                return Err(validation_err);
            }

            if attempt == 1 {
                info!(steps = plan.steps.len(), "planner plan ready after repair");
            } else {
                info!(steps = plan.steps.len(), "planner plan ready");
            }
            return Ok(plan);
        }

        Err(anyhow::anyhow!("planner retry loop exhausted"))
    }

    async fn repair_plan_response(
        &self,
        goal: &str,
        codebase_summary: &str,
        invalid_raw: &str,
        validation_error: &str,
        config: &CompletionConfig,
        compact_mode: bool,
    ) -> Result<String> {
        let summary_excerpt = truncate_for_prompt(codebase_summary, 6_000);
        let invalid_excerpt = truncate_for_prompt(invalid_raw, 6_000);
        let messages = vec![
            Message {
                role: Role::System,
                content: "You repair invalid software plans. Return only JSON.".to_string(),
            },
            Message {
                role: Role::User,
                content: format!(
                    "Goal:\n{goal}\n\nCodebase summary (excerpt):\n{summary_excerpt}\n\nValidation error:\n{validation_error}\n\nInvalid plan JSON (excerpt):\n{invalid_excerpt}\n\nReturn ONLY a JSON array of steps.\nRules:\n- Every step must include non-empty target_files.\n- target_files must be relative repository paths only (no absolute paths, no '..').\n- Do not use placeholders like <...>.\n- Preserve original intent and dependencies where possible.\n- Keep schema fields: step, task, target_files, depends_on; optional: hypothesis, predicted_consequences, experiment, confidence.\n{}",
                    if compact_mode {
                        "- Keep output compact: at most 6 steps.\n- Keep each task under 140 characters.\n- Keep target_files short (1-4 concrete paths per step).\n- Omit optional fields unless essential."
                    } else {
                        ""
                    }
                )
            },
        ];

        self.llm.complete(&messages, config).await
    }
}

fn truncate_for_prompt(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let truncated: String = input.chars().take(max_chars).collect();
    format!("{truncated}\n...[truncated]")
}

#[derive(Debug, Deserialize)]
struct JsonPlanStep {
    #[serde(deserialize_with = "deserialize_u32")]
    step: u32,
    task: String,
    #[serde(default, deserialize_with = "deserialize_string_vec")]
    target_files: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_u32_vec")]
    depends_on: Vec<u32>,
    #[serde(default)]
    hypothesis: Option<String>,
    #[serde(default, deserialize_with = "deserialize_string_vec")]
    predicted_consequences: Vec<String>,
    #[serde(default)]
    experiment: Option<ExperimentSpec>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrManyString {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum U32OrString {
    Number(u32),
    Text(String),
}

fn parse_u32_from_text(value: &str) -> Option<u32> {
    let digits = value
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<u32>().ok()
    }
}

fn deserialize_u32<'de, D>(deserializer: D) -> std::result::Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    let value = U32OrString::deserialize(deserializer)?;
    Ok(match value {
        U32OrString::Number(v) => v,
        U32OrString::Text(text) => parse_u32_from_text(&text).unwrap_or(0),
    })
}

fn deserialize_string_vec<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<OneOrManyString>::deserialize(deserializer)?;
    Ok(match value {
        None => Vec::new(),
        Some(OneOrManyString::Many(values)) => values,
        Some(OneOrManyString::One(value)) => {
            if value.trim().is_empty() {
                Vec::new()
            } else {
                vec![value]
            }
        }
    })
}

fn deserialize_u32_vec<'de, D>(deserializer: D) -> std::result::Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = serde_json::Value::deserialize(deserializer)?;
    Ok(match raw {
        serde_json::Value::Null => Vec::new(),
        serde_json::Value::Number(n) => n
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .into_iter()
            .collect(),
        serde_json::Value::String(text) => parse_u32_from_text(&text).into_iter().collect(),
        serde_json::Value::Array(values) => values
            .into_iter()
            .filter_map(|v| match v {
                serde_json::Value::Number(n) => n.as_u64().and_then(|x| u32::try_from(x).ok()),
                serde_json::Value::String(text) => parse_u32_from_text(&text),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    })
}

pub fn parse_plan_response(goal: &str, raw: &str) -> Result<Plan> {
    if let Ok(steps) = serde_json::from_str::<Vec<JsonPlanStep>>(raw) {
        return Ok(Plan {
            goal: goal.to_string(),
            steps: steps.into_iter().map(to_plan_step).collect(),
        });
    }

    #[derive(Deserialize)]
    struct WrappedPlan {
        goal: Option<String>,
        steps: Vec<JsonPlanStep>,
    }

    if let Ok(wrapped) = serde_json::from_str::<WrappedPlan>(raw) {
        return Ok(Plan {
            goal: wrapped.goal.unwrap_or_else(|| goal.to_string()),
            steps: wrapped.steps.into_iter().map(to_plan_step).collect(),
        });
    }

    let normalized = strip_code_fences(raw);
    if let Ok(steps) = serde_json::from_str::<Vec<JsonPlanStep>>(&normalized) {
        return Ok(Plan {
            goal: goal.to_string(),
            steps: steps.into_iter().map(to_plan_step).collect(),
        });
    }
    if let Ok(wrapped) = serde_json::from_str::<WrappedPlan>(&normalized) {
        return Ok(Plan {
            goal: wrapped.goal.unwrap_or_else(|| goal.to_string()),
            steps: wrapped.steps.into_iter().map(to_plan_step).collect(),
        });
    }

    if let Some(extracted) = extract_json_payload(&normalized) {
        if let Ok(steps) = serde_json::from_str::<Vec<JsonPlanStep>>(&extracted) {
            return Ok(Plan {
                goal: goal.to_string(),
                steps: steps.into_iter().map(to_plan_step).collect(),
            });
        }
        if let Ok(wrapped) = serde_json::from_str::<WrappedPlan>(&extracted) {
            return Ok(Plan {
                goal: wrapped.goal.unwrap_or_else(|| goal.to_string()),
                steps: wrapped.steps.into_iter().map(to_plan_step).collect(),
            });
        }
    }

    let wrapped: WrappedPlan = serde_json::from_str(raw)
        .with_context(|| "failed to parse planner response as JSON array or object")?;
    Ok(Plan {
        goal: wrapped.goal.unwrap_or_else(|| goal.to_string()),
        steps: wrapped.steps.into_iter().map(to_plan_step).collect(),
    })
}

fn strip_code_fences(raw: &str) -> String {
    let trimmed = raw.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }

    let mut lines = trimmed.lines();
    let _ = lines.next();
    let mut body = lines.collect::<Vec<_>>();
    if body
        .last()
        .is_some_and(|line| line.trim_start().starts_with("```"))
    {
        body.pop();
    }
    body.join("\n").trim().to_string()
}

fn extract_json_payload(raw: &str) -> Option<String> {
    let s = raw.trim();
    let mut start = None;
    let mut stack: Vec<char> = Vec::new();

    for (i, ch) in s.char_indices() {
        if start.is_none() {
            if ch == '{' || ch == '[' {
                start = Some(i);
                stack.push(ch);
            }
            continue;
        }

        match ch {
            '{' | '[' => stack.push(ch),
            '}' => {
                if stack.last() == Some(&'{') {
                    stack.pop();
                }
            }
            ']' => {
                if stack.last() == Some(&'[') {
                    stack.pop();
                }
            }
            _ => {}
        }

        if stack.is_empty() {
            if let Some(st) = start {
                return Some(s[st..=i].to_string());
            }
        }
    }

    None
}

fn to_plan_step(js: JsonPlanStep) -> PlanStep {
    PlanStep {
        step: js.step,
        task: js.task,
        target_files: js.target_files.into_iter().map(PathBuf::from).collect(),
        depends_on: js.depends_on,
        hypothesis: js.hypothesis,
        predicted_consequences: js.predicted_consequences,
        experiment: js.experiment,
        confidence: js.confidence,
    }
}

fn normalize_step_numbers(plan: &mut Plan) {
    for (idx, step) in plan.steps.iter_mut().enumerate() {
        if step.step == 0 {
            step.step = idx as u32 + 1;
        }
    }
}

pub fn enforce_plan_limits(plan: &mut Plan, max_steps: u32) {
    if max_steps == 0 {
        return;
    }

    if plan.steps.len() <= max_steps as usize {
        return;
    }

    plan.steps.sort_by_key(|s| s.step);
    plan.steps.truncate(max_steps as usize);

    let remap: HashMap<u32, u32> = plan
        .steps
        .iter()
        .enumerate()
        .map(|(idx, step)| (step.step, idx as u32 + 1))
        .collect();

    for step in &mut plan.steps {
        step.depends_on = step
            .depends_on
            .iter()
            .filter_map(|dep| remap.get(dep).copied())
            .collect();
        step.depends_on.sort_unstable();
        step.depends_on.dedup();
    }

    for (idx, step) in plan.steps.iter_mut().enumerate() {
        step.step = idx as u32 + 1;
    }
}

pub fn validate_plan(plan: &Plan) -> Result<()> {
    ensure!(!plan.goal.trim().is_empty(), "plan.goal must not be empty");
    ensure!(!plan.steps.is_empty(), "plan must have at least one step");

    let steps_by_id: HashMap<u32, &PlanStep> = plan.steps.iter().map(|s| (s.step, s)).collect();
    ensure!(
        steps_by_id.len() == plan.steps.len(),
        "plan has duplicate step numbers"
    );

    for step in &plan.steps {
        ensure!(step.step > 0, "step number must be > 0");
        ensure!(
            !step.task.trim().is_empty(),
            "step {} has empty task",
            step.step
        );
        ensure!(
            !step.target_files.is_empty(),
            "step {} must include target_files",
            step.step
        );

        if let Some(hypothesis) = &step.hypothesis {
            ensure!(
                !hypothesis.trim().is_empty(),
                "step {} has empty hypothesis",
                step.step
            );
        }

        if let Some(experiment) = &step.experiment {
            ensure!(
                !experiment.command.trim().is_empty(),
                "step {} has experiment with empty command",
                step.step
            );
        }

        if let Some(confidence) = step.confidence {
            ensure!(
                (0.0..=1.0).contains(&confidence),
                "step {} confidence must be in [0.0, 1.0]",
                step.step
            );
        }

        for file in &step.target_files {
            validate_path(file)
                .with_context(|| format!("invalid target file in step {}", step.step))?;
        }

        for dep in &step.depends_on {
            ensure!(
                steps_by_id.contains_key(dep),
                "step {} depends on missing step {}",
                step.step,
                dep
            );
            ensure!(
                *dep != step.step,
                "step {} cannot depend on itself",
                step.step
            );
        }
    }

    ensure!(!has_cycles(plan), "plan dependency graph has cycles");
    Ok(())
}

fn validate_path(path: &std::path::Path) -> Result<()> {
    let path_str = path.to_string_lossy();
    ensure!(!path.is_absolute(), "absolute paths are not allowed");
    ensure!(!path_str.contains(".."), "parent traversal is not allowed");
    ensure!(!path_str.trim().is_empty(), "empty target path");
    Ok(())
}

fn has_cycles(plan: &Plan) -> bool {
    let graph: HashMap<u32, Vec<u32>> = plan
        .steps
        .iter()
        .map(|step| (step.step, step.depends_on.clone()))
        .collect();

    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();

    fn dfs(
        node: u32,
        graph: &HashMap<u32, Vec<u32>>,
        visiting: &mut HashSet<u32>,
        visited: &mut HashSet<u32>,
    ) -> bool {
        if visited.contains(&node) {
            return false;
        }
        if !visiting.insert(node) {
            return true;
        }

        if let Some(neighbors) = graph.get(&node) {
            for neighbor in neighbors {
                if dfs(*neighbor, graph, visiting, visited) {
                    return true;
                }
            }
        }

        visiting.remove(&node);
        visited.insert(node);
        false
    }

    for node in graph.keys().copied() {
        if dfs(node, &graph, &mut visiting, &mut visited) {
            return true;
        }
    }

    false
}
