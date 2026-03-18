use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use warlock::agents::planner::{enforce_plan_limits, parse_plan_response, validate_plan, Planner};
use warlock::llm::provider::{CompletionConfig, LlmProvider, Message};

struct ScriptedLlm {
    outputs: Mutex<Vec<String>>,
}

#[async_trait]
impl LlmProvider for ScriptedLlm {
    async fn complete(
        &self,
        _messages: &[Message],
        _config: &CompletionConfig,
    ) -> anyhow::Result<String> {
        let mut guard = self.outputs.lock().expect("lock outputs");
        if guard.is_empty() {
            return Err(anyhow::anyhow!("no scripted output available"));
        }
        Ok(guard.remove(0))
    }

    fn provider_name(&self) -> &'static str {
        "scripted"
    }
}

#[test]
fn parses_plan_from_json_array() {
    let raw = r#"
        [
            {
                "step": 1,
                "task": "Update API",
                "target_files": ["src/api.rs"],
                "depends_on": []
            },
            {
                "step": 2,
                "task": "Add tests",
                "target_files": ["tests/api_test.rs"],
                "depends_on": [1]
            }
        ]
        "#;

    let plan = parse_plan_response("improve api", raw).expect("should parse array response");
    assert_eq!(plan.goal, "improve api");
    assert_eq!(plan.steps.len(), 2);
    validate_plan(&plan).expect("parsed plan should validate");
}

#[test]
fn parses_wrapped_plan_object() {
    let raw = r#"
        {
            "goal": "add billing",
            "steps": [
                {
                    "step": 1,
                    "task": "create billing module",
                    "target_files": ["src/billing/mod.rs"],
                    "depends_on": []
                }
            ]
        }
        "#;

    let plan = parse_plan_response("fallback goal", raw).expect("should parse wrapped response");
    assert_eq!(plan.goal, "add billing");
    assert_eq!(plan.steps[0].task, "create billing module");
}

#[test]
fn rejects_cyclic_dependencies() {
    let raw = r#"
        [
            {
                "step": 1,
                "task": "first",
                "target_files": ["src/a.rs"],
                "depends_on": [2]
            },
            {
                "step": 2,
                "task": "second",
                "target_files": ["src/b.rs"],
                "depends_on": [1]
            }
        ]
        "#;

    let plan = parse_plan_response("cycle", raw).expect("parse should succeed");
    let err = validate_plan(&plan).expect_err("validation should fail on cycle");
    assert!(err.to_string().contains("cycles"));
}

#[test]
fn enforces_max_steps_and_remaps_dependencies() {
    let raw = r#"
        [
            {
                "step": 10,
                "task": "first",
                "target_files": ["src/a.rs"],
                "depends_on": []
            },
            {
                "step": 20,
                "task": "second",
                "target_files": ["src/b.rs"],
                "depends_on": [10]
            },
            {
                "step": 30,
                "task": "third",
                "target_files": ["src/c.rs"],
                "depends_on": [20]
            }
        ]
        "#;

    let mut plan = parse_plan_response("trim", raw).expect("parse should succeed");
    enforce_plan_limits(&mut plan, 2);

    assert_eq!(plan.steps.len(), 2);
    assert_eq!(plan.steps[0].step, 1);
    assert_eq!(plan.steps[1].step, 2);
    assert_eq!(plan.steps[1].depends_on, vec![1]);
    validate_plan(&plan).expect("trimmed plan should remain valid");
}

#[test]
fn parses_plan_when_predicted_consequences_is_string() {
    let raw = r#"
        [
            {
                "step": 1,
                "task": "first",
                "target_files": ["src/a.rs"],
                "depends_on": [],
                "predicted_consequences": "single consequence"
            }
        ]
        "#;

    let plan = parse_plan_response("goal", raw).expect("parse should succeed");
    assert_eq!(
        plan.steps[0].predicted_consequences,
        vec!["single consequence".to_string()]
    );
    validate_plan(&plan).expect("plan should validate");
}

#[test]
fn parses_plan_with_string_step_identifiers() {
    let raw = r#"
        [
            {
                "step": "1-add-module-docs",
                "task": "first",
                "target_files": ["src/a.rs"],
                "depends_on": []
            },
            {
                "step": "2-generate-summary",
                "task": "second",
                "target_files": ["src/b.rs"],
                "depends_on": ["1"]
            }
        ]
        "#;

    let plan = parse_plan_response("goal", raw).expect("parse should succeed");
    assert_eq!(plan.steps[0].step, 1);
    assert_eq!(plan.steps[1].step, 2);
    assert_eq!(plan.steps[1].depends_on, vec![1]);
    validate_plan(&plan).expect("plan should validate");
}

#[tokio::test]
async fn planner_repairs_empty_target_files() {
    let llm = Arc::new(ScriptedLlm {
        outputs: Mutex::new(vec![
            r#"[
                {
                    "step": 1,
                    "task": "Audit summaries",
                    "target_files": [],
                    "depends_on": []
                }
            ]"#
            .to_string(),
            r#"[
                {
                    "step": 1,
                    "task": "Audit summaries",
                    "target_files": ["src/indexing/mod.rs"],
                    "depends_on": []
                }
            ]"#
            .to_string(),
        ]),
    });

    let planner = Planner::new(
        llm,
        CompletionConfig {
            model: "mock".to_string(),
            temperature: 0.0,
            max_tokens: 512,
            json_mode: true,
        },
    );

    let plan = planner
        .generate_plan_with_limit("Improve summaries", "Codebase summary", Some(5))
        .await
        .expect("planner should repair invalid target_files");

    assert_eq!(plan.steps.len(), 1);
    assert_eq!(
        plan.steps[0].target_files[0].to_string_lossy(),
        "src/indexing/mod.rs"
    );
}

#[tokio::test]
async fn planner_repairs_empty_response_parse_failure() {
    let llm = Arc::new(ScriptedLlm {
        outputs: Mutex::new(vec![
            "".to_string(),
            r#"[
                {
                    "step": 1,
                    "task": "Refine summary behavior",
                    "target_files": ["src/indexing/mod.rs"],
                    "depends_on": []
                }
            ]"#
            .to_string(),
        ]),
    });

    let planner = Planner::new(
        llm,
        CompletionConfig {
            model: "mock".to_string(),
            temperature: 0.0,
            max_tokens: 512,
            json_mode: true,
        },
    );

    let plan = planner
        .generate_plan_with_limit("Improve summaries", "Codebase summary", Some(5))
        .await
        .expect("planner should repair empty response parse failure");

    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].task, "Refine summary behavior");
}
