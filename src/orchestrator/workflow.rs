use crate::agents::coder::{CodePatch, Coder};
use crate::agents::planner::{validate_plan, Plan, PlanStep, Planner};
use crate::agents::reflector::{ReflectionOutcome, Reflector};
use crate::config::{Config, SandboxBackend};
use crate::embeddings::cache::EmbeddingCache;
use crate::embeddings::embedder::Embedder;
use crate::indexing::{
    build_file_summaries, build_planning_summary, build_symbol_summaries, index_repo_async,
    CodeChunk,
};
use crate::patch::apply::apply_patch;
use crate::patch::revert::revert_applied_patches;
use crate::retrieval::vector_store::{derive_repo_id, SearchFilter, VectorDocType, VectorStore};
use crate::retrieval::{hybrid_rank_hits, rewrite_query};
use crate::sandbox::docker::Sandbox;
use crate::sandbox::local::LocalSandbox;
use crate::sandbox::SandboxRunner;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveReport {
    pub goal: String,
    pub plan: Option<Plan>,
    pub step_results: Vec<StepResult>,
    pub overall_success: bool,
    pub duration_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StepResult {
    pub step: u32,
    pub task: String,
    pub status: StepStatus,
    pub error: Option<String>,
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment: Option<ExperimentOutcome>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentOutcome {
    pub hypothesis: String,
    pub command: String,
    pub passed: bool,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Succeeded,
    Failed,
    Skipped,
}

const STAGE_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum StageCheckpointStatus {
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StageCheckpoint {
    stage_id: String,
    title: String,
    status: StageCheckpointStatus,
    attempts: u32,
    started_at: u64,
    updated_at: u64,
    finished_at: Option<u64>,
    markdown_path: String,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SolveRunState {
    schema_version: u32,
    run_id: String,
    repo_id: String,
    goal: String,
    goal_hash: String,
    status: StageCheckpointStatus,
    started_at: u64,
    updated_at: u64,
    finished_at: Option<u64>,
    stages: Vec<StageCheckpoint>,
}

struct SolveStageTracker {
    run_dir: PathBuf,
    state_path: PathBuf,
    state: SolveRunState,
}

impl SolveStageTracker {
    fn load_or_create(repo_root: &Path, repo_id: &str, goal: &str) -> Result<Self> {
        let goal_hash = goal_hash(goal);
        let run_id = format!("{}_{}", now_unix_secs(), &goal_hash[..12]);
        let root = repo_root.join(".warlock").join("runs").join("checkpoints");
        let run_dir = root.join(format!("{}_{}", sanitize_for_filename(repo_id), goal_hash));
        let stages_dir = run_dir.join("stages");
        let state_path = run_dir.join("state.json");

        fs::create_dir_all(&stages_dir).with_context(|| {
            format!("failed to create run checkpoint dir {}", run_dir.display())
        })?;

        let state = if state_path.exists() {
            let raw = fs::read_to_string(&state_path)
                .with_context(|| format!("failed reading state file {}", state_path.display()))?;
            let mut loaded: SolveRunState = serde_json::from_str(&raw)
                .with_context(|| format!("failed parsing state file {}", state_path.display()))?;

            if loaded.schema_version != STAGE_STATE_SCHEMA_VERSION
                || loaded.goal_hash != goal_hash
                || loaded.repo_id != repo_id
            {
                Self::new_state(run_id, repo_id, goal, goal_hash)
            } else {
                loaded.updated_at = now_unix_secs();
                loaded.status = StageCheckpointStatus::InProgress;
                loaded.finished_at = None;
                loaded
            }
        } else {
            Self::new_state(run_id, repo_id, goal, goal_hash)
        };

        let tracker = Self {
            run_dir,
            state_path,
            state,
        };
        tracker.persist_state()?;
        Ok(tracker)
    }

    fn begin_stage(&mut self, stage_id: &str, title: &str, details: Option<&str>) -> Result<()> {
        let now = now_unix_secs();
        let file_name = sanitize_for_filename(stage_id);

        let stage_snapshot = if let Some(existing) = self
            .state
            .stages
            .iter_mut()
            .find(|stage| stage.stage_id == stage_id)
        {
            existing.attempts += 1;
            existing.title = title.to_string();
            existing.status = StageCheckpointStatus::InProgress;
            existing.started_at = now;
            existing.updated_at = now;
            existing.finished_at = None;
            existing.error = None;
            existing.markdown_path =
                format!("stages/{}_attempt_{}.md", file_name, existing.attempts);
            existing.clone()
        } else {
            self.state.stages.push(StageCheckpoint {
                stage_id: stage_id.to_string(),
                title: title.to_string(),
                status: StageCheckpointStatus::InProgress,
                attempts: 1,
                started_at: now,
                updated_at: now,
                finished_at: None,
                markdown_path: format!("stages/{}_attempt_1.md", file_name),
                error: None,
            });
            self.state
                .stages
                .iter_mut()
                .find(|stage| stage.stage_id == stage_id)
                .context("failed to create stage checkpoint")?
                .clone()
        };

        self.state.status = StageCheckpointStatus::InProgress;
        self.state.updated_at = now;
        let markdown_path = self.run_dir.join(&stage_snapshot.markdown_path);
        self.write_stage_markdown(&stage_snapshot, details, None)?;
        self.persist_state()?;

        if !markdown_path.exists() {
            return Err(anyhow::anyhow!(
                "failed to create stage markdown {}",
                markdown_path.display()
            ));
        }

        Ok(())
    }

    fn complete_stage(&mut self, stage_id: &str, details: Option<&str>) -> Result<()> {
        self.update_stage(stage_id, StageCheckpointStatus::Completed, None, details)
    }

    fn fail_stage(&mut self, stage_id: &str, error: &str, details: Option<&str>) -> Result<()> {
        self.update_stage(
            stage_id,
            StageCheckpointStatus::Failed,
            Some(error.to_string()),
            details,
        )
    }

    fn finish_run(&mut self, ok: bool) -> Result<()> {
        let now = now_unix_secs();
        self.state.status = if ok {
            StageCheckpointStatus::Completed
        } else {
            StageCheckpointStatus::Failed
        };
        self.state.updated_at = now;
        self.state.finished_at = Some(now);
        self.persist_state()
    }

    fn run_id(&self) -> &str {
        &self.state.run_id
    }

    fn update_stage(
        &mut self,
        stage_id: &str,
        status: StageCheckpointStatus,
        error: Option<String>,
        details: Option<&str>,
    ) -> Result<()> {
        let stage_snapshot = {
            let stage = self
                .state
                .stages
                .iter_mut()
                .find(|stage| stage.stage_id == stage_id)
                .with_context(|| format!("unknown stage id: {stage_id}"))?;

            let now = now_unix_secs();
            stage.status = status;
            stage.updated_at = now;
            stage.finished_at = Some(now);
            stage.error = error;
            self.state.updated_at = now;
            stage.clone()
        };

        self.write_stage_markdown(&stage_snapshot, details, stage_snapshot.error.as_deref())?;
        self.persist_state()
    }

    fn persist_state(&self) -> Result<()> {
        let serialized = serde_json::to_string_pretty(&self.state)?;
        fs::write(&self.state_path, serialized)
            .with_context(|| format!("failed writing state file {}", self.state_path.display()))
    }

    fn write_stage_markdown(
        &self,
        stage: &StageCheckpoint,
        details: Option<&str>,
        error: Option<&str>,
    ) -> Result<()> {
        let path = self.run_dir.join(&stage.markdown_path);
        let mut md = String::new();
        md.push_str("---\n");
        md.push_str(&format!("schema_version: {}\n", STAGE_STATE_SCHEMA_VERSION));
        md.push_str(&format!("run_id: {}\n", self.state.run_id));
        md.push_str(&format!("stage_id: {}\n", stage.stage_id));
        md.push_str(&format!("title: {}\n", escape_yaml_scalar(&stage.title)));
        md.push_str(&format!("status: {:?}\n", stage.status));
        md.push_str(&format!("attempt: {}\n", stage.attempts));
        md.push_str(&format!("started_at: {}\n", stage.started_at));
        md.push_str(&format!("updated_at: {}\n", stage.updated_at));
        if let Some(finished_at) = stage.finished_at {
            md.push_str(&format!("finished_at: {}\n", finished_at));
        }
        md.push_str("---\n\n");
        md.push_str(&format!("# Stage {}\n\n", stage.stage_id));
        md.push_str(&format!("- Title: {}\n", stage.title));
        md.push_str(&format!("- Status: {:?}\n", stage.status));
        md.push_str(&format!("- Attempt: {}\n\n", stage.attempts));

        md.push_str("## Details\n\n");
        md.push_str(details.unwrap_or("(none)"));
        md.push_str("\n\n");

        if let Some(err) = error {
            md.push_str("## Error\n\n");
            md.push_str(err);
            md.push('\n');
        }

        fs::write(&path, md)
            .with_context(|| format!("failed writing stage markdown {}", path.display()))
    }

    fn new_state(run_id: String, repo_id: &str, goal: &str, goal_hash: String) -> SolveRunState {
        let now = now_unix_secs();
        SolveRunState {
            schema_version: STAGE_STATE_SCHEMA_VERSION,
            run_id,
            repo_id: repo_id.to_string(),
            goal: goal.to_string(),
            goal_hash,
            status: StageCheckpointStatus::InProgress,
            started_at: now,
            updated_at: now,
            finished_at: None,
            stages: Vec::new(),
        }
    }
}

pub struct Orchestrator {
    config: Config,
    repo_path: PathBuf,
    repo_id: String,
    embedder: Embedder,
    embedding_cache: EmbeddingCache,
    store: VectorStore,
    planner: Planner,
    coder: Coder,
    reflector: Reflector,
    sandbox: Box<dyn SandboxRunner>,
}

impl Orchestrator {
    pub async fn new(
        config: Config,
        repo_path: PathBuf,
        planner: Planner,
        coder: Coder,
        reflector: Reflector,
    ) -> Result<Self> {
        let repo_id = derive_repo_id(&repo_path);
        let embedder = Embedder::new(&config.embeddings)?;
        let embedding_cache = if config.embeddings.persist_cache {
            let db_path = config
                .embeddings
                .cache_db_path
                .as_ref()
                .map(PathBuf::from)
                .map(|p| {
                    if p.is_absolute() {
                        p
                    } else {
                        repo_path.join(p)
                    }
                })
                .unwrap_or_else(|| repo_path.join(".warlock/embeddings_cache"));
            EmbeddingCache::with_db(&db_path)?
        } else {
            EmbeddingCache::new()
        };
        let store = VectorStore::new(
            &config.vector_store.url,
            &config.vector_store.collection,
            config.embeddings.dimensions as u64,
        )
        .await?;
        let sandbox: Box<dyn SandboxRunner> = match config.sandbox.backend {
            SandboxBackend::Local => Box::new(LocalSandbox::new(&config.sandbox)),
            SandboxBackend::Docker => Box::new(Sandbox::new(&config.sandbox)?),
        };

        Ok(Self {
            config,
            repo_path,
            repo_id,
            embedder,
            embedding_cache,
            store,
            planner,
            coder,
            reflector,
            sandbox,
        })
    }

    pub async fn solve(&mut self, goal: &str) -> Result<SolveReport> {
        let started = Instant::now();
        let mut stage_tracker =
            SolveStageTracker::load_or_create(&self.repo_path, &self.repo_id, goal)?;
        info!(
            goal_len = goal.len(),
            max_plan_cycles = self.config.agent.max_plan_cycles,
            run_id = stage_tracker.run_id(),
            "solve start"
        );

        let mut report = SolveReport {
            goal: goal.to_string(),
            plan: None,
            step_results: Vec::new(),
            overall_success: true,
            duration_secs: 0.0,
        };

        let mut plan_feedback = String::new();
        for cycle in 1..=self.config.agent.max_plan_cycles {
            info!(cycle, "solve cycle start");
            let index_stage = format!("cycle_{cycle}.index_context");
            stage_tracker.begin_stage(
                &index_stage,
                "Index repository and upsert context",
                Some("Builds code chunks and refreshes code/file/symbol context vectors."),
            )?;
            let chunks = match index_repo_async(&self.repo_path, &self.config.indexing).await {
                Ok(chunks) => chunks,
                Err(err) => {
                    stage_tracker.fail_stage(
                        &index_stage,
                        &err.to_string(),
                        Some("Repository indexing failed before context upsert."),
                    )?;
                    stage_tracker.finish_run(false)?;
                    return Err(err);
                }
            };
            debug!(
                cycle,
                chunks = chunks.len(),
                "solve cycle indexed repository"
            );
            if let Err(err) = self.upsert_all_context(&chunks).await {
                stage_tracker.fail_stage(
                    &index_stage,
                    &err.to_string(),
                    Some("Vector context upsert failed."),
                )?;
                stage_tracker.finish_run(false)?;
                return Err(err);
            }
            stage_tracker.complete_stage(
                &index_stage,
                Some(&format!(
                    "Indexed {} chunk(s) and refreshed context vectors.",
                    chunks.len()
                )),
            )?;

            let summary = build_planning_summary(&chunks);
            let planning_goal = if plan_feedback.is_empty() {
                goal.to_string()
            } else {
                format!(
                    "{}\n\nPrevious cycle failures:\n{}\n\nCreate an improved plan that addresses these failures.",
                    goal, plan_feedback
                )
            };

            let plan_stage = format!("cycle_{cycle}.generate_plan");
            stage_tracker.begin_stage(
                &plan_stage,
                "Generate execution plan",
                Some("Planner creates and validates a bounded step plan."),
            )?;
            let plan = self
                .planner
                .generate_plan_with_limit(
                    &planning_goal,
                    &summary,
                    Some(self.config.agent.planner_max_steps),
                )
                .await;
            let plan = match plan {
                Ok(plan) => plan,
                Err(err) => {
                    stage_tracker.fail_stage(
                        &plan_stage,
                        &err.to_string(),
                        Some("Planner request failed."),
                    )?;
                    stage_tracker.finish_run(false)?;
                    return Err(err);
                }
            };
            if let Err(err) = validate_plan(&plan) {
                stage_tracker.fail_stage(
                    &plan_stage,
                    &err.to_string(),
                    Some("Plan validation failed."),
                )?;
                stage_tracker.finish_run(false)?;
                return Err(err);
            }
            stage_tracker.complete_stage(
                &plan_stage,
                Some(&format!(
                    "Generated plan with {} step(s).",
                    plan.steps.len()
                )),
            )?;
            info!(
                cycle,
                steps = plan.steps.len(),
                "solve cycle plan generated"
            );
            report.plan = Some(plan.clone());

            let cycle_stage = format!("cycle_{cycle}.execute_plan");
            stage_tracker.begin_stage(
                &cycle_stage,
                "Execute plan cycle",
                Some("Runs experiments, code generation, validation and reflection per step."),
            )?;
            let cycle_success = match self
                .run_plan_cycle(
                    cycle,
                    &plan,
                    goal,
                    &summary,
                    &mut report,
                    &mut stage_tracker,
                )
                .await
            {
                Ok(success) => success,
                Err(err) => {
                    stage_tracker.fail_stage(
                        &cycle_stage,
                        &err.to_string(),
                        Some("Plan cycle execution terminated with an error."),
                    )?;
                    stage_tracker.finish_run(false)?;
                    return Err(err);
                }
            };
            stage_tracker.complete_stage(
                &cycle_stage,
                Some(&format!("Cycle success: {}", cycle_success)),
            )?;
            if cycle_success {
                report.overall_success = true;
                info!(cycle, "solve cycle succeeded");
                break;
            }

            report.overall_success = false;
            plan_feedback = summarize_recent_failures(&report.step_results, cycle as usize);
            warn!(cycle, "solve cycle failed; preparing replanning");
        }

        report.duration_secs = started.elapsed().as_secs_f64();
        info!(
            overall_success = report.overall_success,
            duration_secs = report.duration_secs,
            "solve end"
        );
        stage_tracker.finish_run(report.overall_success)?;
        Ok(report)
    }

    async fn upsert_chunks(&mut self, chunks: &[CodeChunk]) -> Result<()> {
        let embeddings = self
            .embedder
            .embed_chunks_with_cache(chunks, &self.embedding_cache)
            .await?;
        self.store.upsert(&self.repo_id, chunks, &embeddings).await
    }

    async fn upsert_all_context(&mut self, chunks: &[CodeChunk]) -> Result<()> {
        self.upsert_chunks(chunks).await?;

        let file_summaries = build_file_summaries(chunks);
        let symbol_summaries = build_symbol_summaries(chunks);

        let file_task = async {
            if file_summaries.is_empty() {
                return Ok(());
            }

            let texts = file_summaries
                .iter()
                .map(|s| s.summary.clone())
                .collect::<Vec<_>>();
            let embeddings = self.embedder.embed_batch_chunked(&texts).await?;
            self.store
                .upsert_file_summaries(&self.repo_id, &file_summaries, &embeddings)
                .await
        };

        let symbol_task = async {
            if symbol_summaries.is_empty() {
                return Ok(());
            }

            let texts = symbol_summaries
                .iter()
                .map(|s| s.summary.clone())
                .collect::<Vec<_>>();
            let embeddings = self.embedder.embed_batch_chunked(&texts).await?;
            self.store
                .upsert_symbol_summaries(&self.repo_id, &symbol_summaries, &embeddings)
                .await
        };

        tokio::try_join!(file_task, symbol_task)?;

        Ok(())
    }

    async fn run_plan_cycle(
        &mut self,
        cycle: u32,
        plan: &Plan,
        goal: &str,
        summary: &str,
        report: &mut SolveReport,
        stage_tracker: &mut SolveStageTracker,
    ) -> Result<bool> {
        let mut cycle_success = true;
        let mut step_statuses: HashMap<u32, StepStatus> = HashMap::new();
        info!(steps = plan.steps.len(), "plan cycle execution start");

        for step in &plan.steps {
            let step_started = Instant::now();
            info!(step = step.step, task = %step.task, depends_on = ?step.depends_on, "step start");
            let blocked_by = failed_dependencies(step, &step_statuses);
            if !blocked_by.is_empty() {
                let blocked_display = blocked_by
                    .iter()
                    .map(|dep| dep.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                warn!(step = step.step, "step skipped due to failed dependency");
                report.step_results.push(StepResult {
                    step: step.step,
                    task: step.task.clone(),
                    status: StepStatus::Skipped,
                    error: Some(format!("dependency failed: step(s) {blocked_display}")),
                    attempts: 0,
                    experiment: None,
                });
                step_statuses.insert(step.step, StepStatus::Skipped);
                cycle_success = false;
                continue;
            }

            let experiment_stage = format!("cycle_{cycle}.step_{}.experiment", step.step);
            let experiment_outcome = if self.config.agent.planning_experiments_enabled {
                info!(step = step.step, "planning experiment start");
                stage_tracker.begin_stage(
                    &experiment_stage,
                    &format!("Run experiment for step {}", step.step),
                    Some(&step.task),
                )?;
                match self.run_step_experiment(step).await {
                    Ok(outcome) => {
                        stage_tracker.complete_stage(
                            &experiment_stage,
                            Some(&format!(
                                "passed={}, notes={}",
                                outcome.passed, outcome.notes
                            )),
                        )?;
                        Some(outcome)
                    }
                    Err(err) => {
                        stage_tracker.fail_stage(
                            &experiment_stage,
                            &err.to_string(),
                            Some("Experiment command failed to execute."),
                        )?;
                        return Err(err);
                    }
                }
            } else {
                None
            };

            if let Some(outcome) = &experiment_outcome {
                info!(
                    step = step.step,
                    passed = outcome.passed,
                    "planning experiment completed"
                );
            }

            if self.config.agent.planning_experiment_strict
                && experiment_outcome
                    .as_ref()
                    .is_some_and(|outcome| !outcome.passed)
            {
                let notes = experiment_outcome
                    .as_ref()
                    .map(|e| e.notes.clone())
                    .unwrap_or_else(|| "experiment failed".to_string());
                report.step_results.push(StepResult {
                    step: step.step,
                    task: step.task.clone(),
                    status: StepStatus::Failed,
                    error: Some(format!("planning experiment failed: {notes}")),
                    attempts: 0,
                    experiment: experiment_outcome.clone(),
                });
                step_statuses.insert(step.step, StepStatus::Failed);
                cycle_success = false;
                warn!(step = step.step, notes = %notes, "step blocked by strict planning experiment");
                continue;
            }

            let exec_stage = format!("cycle_{cycle}.step_{}.execute", step.step);
            stage_tracker.begin_stage(
                &exec_stage,
                &format!("Execute step {}", step.step),
                Some(&step.task),
            )?;
            let result = self.execute_step_with_reflection(step).await;
            match result {
                Ok(attempts) => {
                    stage_tracker.complete_stage(
                        &exec_stage,
                        Some(&format!("Step completed in {} attempt(s).", attempts)),
                    )?;
                    report.step_results.push(StepResult {
                        step: step.step,
                        task: step.task.clone(),
                        status: StepStatus::Succeeded,
                        error: None,
                        attempts,
                        experiment: experiment_outcome.clone(),
                    });
                    step_statuses.insert(step.step, StepStatus::Succeeded);
                    let updated = index_repo_async(&self.repo_path, &self.config.indexing).await?;
                    self.upsert_all_context(&updated).await?;
                    info!(
                        step = step.step,
                        attempts,
                        elapsed_ms = step_started.elapsed().as_millis() as u64,
                        "step succeeded"
                    );
                }
                Err(e) => {
                    cycle_success = false;
                    let failure = e.to_string();
                    stage_tracker.fail_stage(
                        &exec_stage,
                        &failure,
                        Some("Primary step execution failed; attempting replanning."),
                    )?;
                    warn!(step = step.step, error = %failure, elapsed_ms = step_started.elapsed().as_millis() as u64, "step failed");
                    if let Some(replanned) = self
                        .replan_failed_step(step, goal, summary, &failure)
                        .await?
                    {
                        info!(step = step.step, "step replanning start");
                        let replan_stage =
                            format!("cycle_{cycle}.step_{}.replan_execute", step.step);
                        stage_tracker.begin_stage(
                            &replan_stage,
                            &format!("Execute replanned step {}", step.step),
                            Some(&replanned.task),
                        )?;
                        match self.execute_step_with_reflection(&replanned).await {
                            Ok(attempts) => {
                                stage_tracker.complete_stage(
                                    &replan_stage,
                                    Some(&format!(
                                        "Replanned step completed in {} attempt(s).",
                                        attempts
                                    )),
                                )?;
                                report.step_results.push(StepResult {
                                    step: step.step,
                                    task: format!("{} [replanned]", step.task),
                                    status: StepStatus::Succeeded,
                                    error: None,
                                    attempts,
                                    experiment: experiment_outcome.clone(),
                                });
                                step_statuses.insert(step.step, StepStatus::Succeeded);
                                let updated =
                                    index_repo_async(&self.repo_path, &self.config.indexing)
                                        .await?;
                                self.upsert_all_context(&updated).await?;
                                info!(step = step.step, attempts, "replanned step succeeded");
                                continue;
                            }
                            Err(replanned_err) => {
                                stage_tracker.fail_stage(
                                    &replan_stage,
                                    &replanned_err.to_string(),
                                    Some("Replanned execution failed."),
                                )?;
                                warn!(step = step.step, error = %replanned_err, "replanned step failed");
                                report.step_results.push(StepResult {
                                    step: step.step,
                                    task: step.task.clone(),
                                    status: StepStatus::Failed,
                                    error: Some(format!(
                                        "original failure: {failure}; replanned failure: {}",
                                        replanned_err
                                    )),
                                    attempts: self.config.agent.max_reflection_attempts,
                                    experiment: experiment_outcome.clone(),
                                });
                                step_statuses.insert(step.step, StepStatus::Failed);
                            }
                        }
                    } else {
                        report.step_results.push(StepResult {
                            step: step.step,
                            task: step.task.clone(),
                            status: StepStatus::Failed,
                            error: Some(failure),
                            attempts: self.config.agent.max_reflection_attempts,
                            experiment: experiment_outcome.clone(),
                        });
                        step_statuses.insert(step.step, StepStatus::Failed);
                    }
                }
            }
        }

        info!(cycle_success, "plan cycle execution end");
        Ok(cycle_success)
    }

    async fn execute_step_with_reflection(&mut self, step: &PlanStep) -> Result<u32> {
        let mut previous_error = String::new();
        let mut attempt_history = Vec::new();
        let retrieval_top_k = std::cmp::max(
            1,
            std::cmp::min(
                self.config.agent.coder_context_chunks as usize,
                self.config.vector_store.top_k as usize,
            ),
        );

        for attempt in 1..=self.config.agent.max_reflection_attempts {
            let query_text = if previous_error.is_empty() {
                step.task.clone()
            } else {
                format!(
                    "{}\n\nPrevious errors:\n{}\n\nAttempt history:\n{}",
                    step.task,
                    previous_error,
                    attempt_history.join("\n")
                )
            };

            let query_variants = rewrite_query(&query_text);
            let query_variants = if query_variants.is_empty() {
                vec![query_text.clone()]
            } else {
                query_variants
            };

            let query_vecs = self.embedder.embed_batch_chunked(&query_variants).await?;

            let mut aggregated_hits = Vec::new();
            for query_vec in &query_vecs {
                let hits = self
                    .store
                    .search_with_scores(&self.repo_id, query_vec.clone(), retrieval_top_k * 2, None)
                    .await?;
                aggregated_hits.extend(hits);
            }

            let mut context_chunks = hybrid_rank_hits(&query_text, aggregated_hits, 0.7)
                .into_iter()
                .map(|h| h.chunk)
                .collect::<Vec<_>>();

            for target in &step.target_files {
                let filter = SearchFilter {
                    file: Some(target.to_string_lossy().to_string()),
                    language: None,
                    symbol_kind: None,
                    doc_type: None,
                };
                for query_vec in &query_vecs {
                    let filtered = self
                        .store
                        .search(
                            &self.repo_id,
                            query_vec.clone(),
                            std::cmp::max(1, retrieval_top_k / 2),
                            Some(&filter),
                        )
                        .await?;
                    context_chunks.extend(filtered);
                }
            }

            context_chunks.sort_by(|a, b| a.file.cmp(&b.file).then_with(|| a.span.cmp(&b.span)));
            context_chunks
                .dedup_by(|a, b| a.file == b.file && a.symbol == b.symbol && a.span == b.span);

            let mut summary_context = Vec::new();
            if let Some(primary_query_vec) = query_vecs.first() {
                let symbol_summaries = self
                    .store
                    .search_summaries_with_scores(
                        &self.repo_id,
                        primary_query_vec.clone(),
                        5,
                        VectorDocType::SymbolSummary,
                    )
                    .await?;
                summary_context.extend(symbol_summaries.into_iter().map(|s| {
                    format!(
                        "[symbol-summary] {}::{} - {}",
                        s.file,
                        s.symbol.unwrap_or_default(),
                        s.summary
                    )
                }));

                let file_summaries = self
                    .store
                    .search_summaries_with_scores(
                        &self.repo_id,
                        primary_query_vec.clone(),
                        2,
                        VectorDocType::FileSummary,
                    )
                    .await?;
                summary_context.extend(
                    file_summaries
                        .into_iter()
                        .map(|s| format!("[file-summary] {} - {}", s.file, s.summary)),
                );
            }

            let mut retrieved = summary_context;
            retrieved.extend(
                context_chunks
                    .into_iter()
                    .take(retrieval_top_k)
                    .map(|c| c.code)
                    .collect::<Vec<_>>(),
            );
            let targets = read_target_files(&self.repo_path, &step.target_files)?;
            let patches = match self
                .coder
                .generate_patches(step, &retrieved, &targets)
                .await
            {
                Ok(patches) => patches,
                Err(err) => {
                    previous_error = format!("coder failed to generate patch: {err:#}");
                    attempt_history.push(format!(
                        "attempt {attempt}: patch generation failed ({})",
                        truncate_for_prompt(&previous_error, 400)
                    ));
                    continue;
                }
            };

            if let Err(prediction_err) = predict_patch_applicability(&self.repo_path, &patches) {
                previous_error = format!("pre-apply prediction failed: {prediction_err}");
                attempt_history.push(format!(
                    "attempt {attempt}: prediction failed ({})",
                    truncate_for_prompt(&previous_error, 400)
                ));
                continue;
            }

            apply_patches(&self.repo_path, &patches)?;
            let compile_result = self
                .sandbox
                .run_command(&self.repo_path, "cargo check --quiet")
                .await?;
            if !compile_result.success() {
                previous_error = format!("compile prediction failed:\n{}", compile_result.stderr);
                attempt_history.push(format!(
                    "attempt {attempt}: cargo check failed ({})",
                    truncate_for_prompt(&compile_result.stderr, 400)
                ));
                revert_applied_patches(&self.repo_path)?;
                continue;
            }

            let test_result = self
                .sandbox
                .run_tests(&self.repo_path, "cargo test --quiet")
                .await?;

            if test_result.success() {
                return Ok(attempt);
            }

            previous_error = test_result.stderr;
            attempt_history.push(format!(
                "attempt {attempt}: tests failed ({})",
                truncate_for_prompt(&previous_error, 400)
            ));
            revert_applied_patches(&self.repo_path)?;

            let failed_diff = patches
                .iter()
                .map(|p| p.diff.clone())
                .collect::<Vec<_>>()
                .join("\n");

            match self
                .reflector
                .reflect(
                    &step.task,
                    &failed_diff,
                    &previous_error,
                    &attempt_history.join("\n"),
                )
                .await?
            {
                ReflectionOutcome::Escalate(reason) => {
                    return Err(anyhow::anyhow!("reflector escalated: {reason}"));
                }
                ReflectionOutcome::Corrected(corrected) => {
                    predict_patch_applicability(&self.repo_path, &corrected)
                        .context("corrected patch failed pre-apply prediction")?;
                    apply_patches(&self.repo_path, &corrected)?;
                    let corrected_compile = self
                        .sandbox
                        .run_command(&self.repo_path, "cargo check --quiet")
                        .await?;
                    if !corrected_compile.success() {
                        previous_error = corrected_compile.stderr;
                        attempt_history.push(format!(
                            "attempt {attempt}: reflected compile failed ({})",
                            truncate_for_prompt(&previous_error, 400)
                        ));
                        revert_applied_patches(&self.repo_path)?;
                        continue;
                    }

                    let second_try = self
                        .sandbox
                        .run_tests(&self.repo_path, "cargo test --quiet")
                        .await?;
                    if second_try.success() {
                        return Ok(attempt);
                    }
                    previous_error = second_try.stderr;
                    attempt_history.push(format!(
                        "attempt {attempt}: reflected tests failed ({})",
                        truncate_for_prompt(&previous_error, 400)
                    ));
                    revert_applied_patches(&self.repo_path)?;
                }
            }
        }

        Err(anyhow::anyhow!(
            "step {} failed after {} attempts: {}",
            step.step,
            self.config.agent.max_reflection_attempts,
            previous_error
        ))
    }

    async fn run_step_experiment(&self, step: &PlanStep) -> Result<ExperimentOutcome> {
        let hypothesis = step.hypothesis.clone().unwrap_or_else(|| {
            format!(
                "Applying step '{}' should improve target behavior while preserving build/test baseline.",
                step.task
            )
        });

        let (command, success_contains) = if let Some(experiment) = &step.experiment {
            (
                experiment.command.trim().to_string(),
                experiment
                    .success_contains
                    .iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>(),
            )
        } else {
            (infer_experiment_command(step), Vec::new())
        };

        let mut actionable_success_contains = success_contains
            .iter()
            .filter(|s| is_actionable_success_marker(s))
            .cloned()
            .collect::<Vec<_>>();
        let ignored_success_markers = success_contains.len() - actionable_success_contains.len();

        info!(step = step.step, command = %command, "experiment command start");
        let mut result = self.sandbox.run_command(&self.repo_path, &command).await?;
        let mut executed_command = command.clone();
        let mut fallback_note = String::new();

        if is_unexecutable_command_failure(&result) {
            let fallback_command = infer_experiment_command(step);
            if fallback_command != command {
                warn!(
                    step = step.step,
                    command = %command,
                    fallback_command = %fallback_command,
                    "experiment command unavailable; retrying with inferred fallback"
                );
                let original_stderr = truncate_for_prompt(result.stderr.trim(), 240);
                result = self
                    .sandbox
                    .run_command(&self.repo_path, &fallback_command)
                    .await?;
                executed_command = fallback_command;
                actionable_success_contains.clear();
                fallback_note = format!(
                    "primary experiment command unavailable; used fallback. original stderr: {}",
                    if original_stderr.is_empty() {
                        "command not found".to_string()
                    } else {
                        original_stderr
                    }
                );
            }
        }

        let merged_output = format!("{}\n{}", result.stdout, result.stderr);
        let missing_checks = actionable_success_contains
            .iter()
            .filter(|needle| !contains_ignore_ascii_case(&merged_output, needle))
            .cloned()
            .collect::<Vec<_>>();
        let passed = result.success() && missing_checks.is_empty();

        let mut notes = if passed {
            "experiment succeeded".to_string()
        } else {
            let mut reasons = Vec::new();
            if !result.success() {
                reasons.push(format!("exit_code={}", result.exit_code));
            }
            if !missing_checks.is_empty() {
                reasons.push(format!(
                    "missing expected output: {}",
                    missing_checks.join(", ")
                ));
            }
            let error_excerpt = truncate_for_prompt(result.stderr.trim(), 300);
            if !error_excerpt.is_empty() {
                reasons.push(format!("stderr: {error_excerpt}"));
            }
            reasons.join("; ")
        };

        if ignored_success_markers > 0 {
            let marker_note = format!(
                "ignored {} non-actionable success marker(s)",
                ignored_success_markers
            );
            notes = if notes.is_empty() {
                marker_note
            } else {
                format!("{notes}; {marker_note}")
            };
        }

        if !fallback_note.is_empty() {
            notes = if notes.is_empty() {
                fallback_note
            } else {
                format!("{notes}; {fallback_note}")
            };
        }

        info!(
            step = step.step,
            passed,
            exit_code = result.exit_code,
            "experiment command completed"
        );

        Ok(ExperimentOutcome {
            hypothesis,
            command: executed_command,
            passed,
            notes,
        })
    }

    async fn replan_failed_step(
        &self,
        step: &PlanStep,
        goal: &str,
        codebase_summary: &str,
        error: &str,
    ) -> Result<Option<PlanStep>> {
        let recovery_goal = format!(
			"Goal: {goal}\n\nRecover failed step:\nStep {} - {}\nTarget files: {:?}\nFailure:\n{}\n\nReturn exactly one revised step.",
			step.step,
			step.task,
			step.target_files,
			truncate_for_prompt(error, 2000)
		);

        let replanned = self
            .planner
            .generate_plan_with_limit(&recovery_goal, codebase_summary, Some(1))
            .await;

        match replanned {
            Ok(plan) => Ok(plan.steps.into_iter().next()),
            Err(_) => Ok(None),
        }
    }
}

fn failed_dependencies(step: &PlanStep, statuses: &HashMap<u32, StepStatus>) -> Vec<u32> {
    step.depends_on
        .iter()
        .copied()
        .filter(|dep| {
            matches!(
                statuses.get(dep),
                Some(StepStatus::Failed | StepStatus::Skipped)
            )
        })
        .collect()
}

fn read_target_files(repo_root: &Path, paths: &[PathBuf]) -> Result<Vec<(PathBuf, String)>> {
    let mut out = Vec::with_capacity(paths.len());
    for rel in paths {
        let abs = repo_root.join(rel);
        let contents = std::fs::read_to_string(&abs).unwrap_or_default();
        out.push((rel.clone(), contents));
    }
    Ok(out)
}

fn apply_patches(repo_root: &Path, patches: &[CodePatch]) -> Result<()> {
    for patch in patches {
        apply_patch(repo_root, &patch.diff)?;
    }
    Ok(())
}

fn predict_patch_applicability(repo_root: &Path, patches: &[CodePatch]) -> Result<()> {
    let mut combined = String::new();
    for patch in patches {
        combined.push_str(&patch.diff);
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
    }

    let check = std::process::Command::new("git")
        .arg("apply")
        .arg("--check")
        .arg("-")
        .current_dir(repo_root)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("failed to spawn `git apply --check` for prediction")?;

    let output = write_stdin_and_wait(check, &combined)?;
    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "patch applicability check failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(())
}

fn truncate_for_prompt(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }
    format!("{}...", &text[..max_len])
}

fn write_stdin_and_wait(
    mut child: std::process::Child,
    input: &str,
) -> Result<std::process::Output> {
    use std::io::Write;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(input.as_bytes())
            .context("failed writing patch to git stdin")?;
    }
    let output = child
        .wait_with_output()
        .context("failed waiting for git apply --check process")?;
    Ok(output)
}

fn summarize_recent_failures(step_results: &[StepResult], take_last: usize) -> String {
    step_results
        .iter()
        .rev()
        .filter(|s| s.status == StepStatus::Failed)
        .take(take_last.max(1))
        .map(|s| {
            format!(
                "Step {} ({}) failed: {}",
                s.step,
                s.task,
                s.error.clone().unwrap_or_else(|| "unknown".to_string())
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn infer_experiment_command(step: &PlanStep) -> String {
    let targets = step
        .target_files
        .iter()
        .map(|p| p.to_string_lossy().to_lowercase())
        .collect::<Vec<_>>();
    let has_test_target = targets.iter().any(|p| {
        p.contains("/tests/")
            || p.starts_with("tests/")
            || p.ends_with("_test.rs")
            || p.ends_with("_tests.rs")
    });

    if has_test_target {
        "cargo test --quiet".to_string()
    } else {
        "cargo check --quiet".to_string()
    }
}

fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

fn is_missing_command_failure(result: &crate::sandbox::TestResult) -> bool {
    result.exit_code == 127
        || contains_ignore_ascii_case(&result.stderr, "command not found")
        || contains_ignore_ascii_case(&result.stderr, "not recognized as an internal")
}

fn is_unexecutable_command_failure(result: &crate::sandbox::TestResult) -> bool {
    is_missing_command_failure(result)
        || (result.exit_code == 2
            && contains_ignore_ascii_case(&result.stderr, "syntax error near unexpected token"))
        || contains_ignore_ascii_case(&result.stderr, "sh: -c: line 0: syntax error")
}

fn is_actionable_success_marker(marker: &str) -> bool {
    let trimmed = marker.trim();
    if trimmed.is_empty() {
        return false;
    }

    let word_count = trimmed.split_whitespace().count();
    let has_shell_wildcards =
        trimmed.contains('*') || trimmed.contains("||") || trimmed.contains(';');
    let has_alnum = trimmed.chars().any(|c| c.is_ascii_alphanumeric());

    has_alnum && word_count <= 6 && trimmed.len() <= 80 && !has_shell_wildcards
}

#[cfg(test)]
mod tests {
    use super::{
        is_actionable_success_marker, is_missing_command_failure, is_unexecutable_command_failure,
    };
    use crate::sandbox::TestResult;

    #[test]
    fn detects_missing_command_failure() {
        let result = TestResult {
            exit_code: 127,
            stdout: String::new(),
            stderr: "sh: rg: command not found".to_string(),
        };
        assert!(is_missing_command_failure(&result));
        assert!(is_unexecutable_command_failure(&result));
    }

    #[test]
    fn detects_shell_syntax_failure_as_unexecutable() {
        let result = TestResult {
            exit_code: 2,
            stdout: String::new(),
            stderr: "sh: -c: line 0: syntax error near unexpected token `('".to_string(),
        };
        assert!(is_unexecutable_command_failure(&result));
    }

    #[test]
    fn filters_non_actionable_success_markers() {
        assert!(is_actionable_success_marker("src/indexing/mod.rs"));
        assert!(is_actionable_success_marker("build_module_summaries"));
        assert!(!is_actionable_success_marker(
            "list of files with module doc comments (if any)"
        ));
        assert!(!is_actionable_success_marker(
            "occurrences of build_module_summaries and related indexing functions in src/indexing"
        ));
    }
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn sanitize_for_filename(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

fn goal_hash(goal: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(goal.as_bytes());
    let digest = hasher.finalize();
    hex::encode(digest)
}

fn escape_yaml_scalar(value: &str) -> String {
    value.replace('\n', " ").replace(':', " -")
}
