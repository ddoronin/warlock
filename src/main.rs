use clap::{Parser, Subcommand, ValueEnum};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};
use warlock::agents::coder::Coder;
use warlock::agents::planner::{Plan, Planner};
use warlock::agents::reflector::Reflector;
use warlock::config::{Config, SandboxBackend};
use warlock::embeddings::cache::EmbeddingCache;
use warlock::embeddings::embedder::Embedder;
use warlock::indexing::manifest::RepositoryManifest;
use warlock::indexing::{
    build_file_summaries, build_planning_summary, build_symbol_summaries, index_files_async,
    index_repo_async, CodeChunk,
};
use warlock::llm::build_provider;
use warlock::llm::provider::CompletionConfig;
use warlock::orchestrator::workflow::{Orchestrator, SolveReport, StepResult, StepStatus};
use warlock::retrieval::vector_store::{derive_repo_id, VectorDocType, VectorStore};
use warlock::retrieval::{hybrid_rank_hits, rewrite_query};

#[derive(Debug, Parser)]
#[command(name = "warlock")]
#[command(about = "Autonomous AI coding agent")]
struct Cli {
    #[arg(long, default_value = "warlock.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run full solve pipeline
    Solve {
        goal: String,
        #[arg(long)]
        repo: PathBuf,
        #[arg(long, default_value_t = false)]
        verbose_planning: bool,
        #[arg(long, value_enum)]
        sandbox_backend: Option<SolveSandboxBackendArg>,
    },
    /// Run indexing only
    Index { repo: PathBuf },
    /// Run indexing for multiple repositories
    IndexMany { repos: Vec<PathBuf> },
    /// Generate plan only
    Plan {
        goal: String,
        #[arg(long)]
        repo: PathBuf,
        #[arg(long, default_value_t = false)]
        verbose_planning: bool,
    },
    /// Search semantically indexed code chunks
    Search {
        query: String,
        #[arg(long)]
        repo: PathBuf,
        #[arg(long, default_value_t = false)]
        with_scores: bool,
        #[arg(long, default_value_t = false)]
        refresh_index: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SolveSandboxBackendArg {
    Local,
    Docker,
}

impl From<SolveSandboxBackendArg> for SandboxBackend {
    fn from(value: SolveSandboxBackendArg) -> Self {
        match value {
            SolveSandboxBackendArg::Local => SandboxBackend::Local,
            SolveSandboxBackendArg::Docker => SandboxBackend::Docker,
        }
    }
}

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .or_else(|_| tracing_subscriber::EnvFilter::try_new("info"))
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    info!("warlock startup");
    let cli = Cli::parse();
    info!(command = ?cli.command, config = %cli.config.display(), "cli parsed");

    let config = match Config::load(&cli.config) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("failed to load config: {err}");
            std::process::exit(1);
        }
    };
    info!(provider = %config.llm.provider, model = %config.llm.model, "config loaded");

    match cli.command {
        Some(Commands::Index { repo }) => {
            info!(repo = %repo.display(), "command index start");
            if let Err(err) = sync_vector_index(&config, &repo, true).await {
                warn!(repo = %repo.display(), error = %err, "command index failed");
                eprintln!("index failed: {err}");
                std::process::exit(1);
            }
            info!(repo = %repo.display(), "command index completed");
        }
        Some(Commands::IndexMany { repos }) => {
            info!(repos = repos.len(), "command index-many start");
            if repos.is_empty() {
                eprintln!("index-many failed: provide at least one repo path");
                std::process::exit(1);
            }
            for repo in repos {
                info!(repo = %repo.display(), "command index-many repo start");
                if let Err(err) = sync_vector_index(&config, &repo, true).await {
                    warn!(repo = %repo.display(), error = %err, "command index-many repo failed");
                    eprintln!("index-many failed for {}: {err}", repo.display());
                    std::process::exit(1);
                }
                info!(repo = %repo.display(), "command index-many repo completed");
            }
        }
        Some(Commands::Plan {
            goal,
            repo,
            verbose_planning,
        }) => {
            info!(repo = %repo.display(), verbose_planning, "command plan start");
            if let Err(err) = run_plan_only(config, repo, &goal, verbose_planning).await {
                warn!(error = %err, "command plan failed");
                eprintln!("plan failed: {err}");
                std::process::exit(1);
            }
            info!("command plan completed");
        }
        Some(Commands::Solve {
            goal,
            repo,
            verbose_planning,
            sandbox_backend,
        }) => {
            info!(repo = %repo.display(), verbose_planning, "command solve start");
            let mut solve_config = config;
            if let Some(backend) = sandbox_backend {
                solve_config.sandbox.backend = backend.into();
            }
            if let Err(err) = run_solve(solve_config, repo, &goal, verbose_planning).await {
                warn!(error = %err, "command solve failed");
                eprintln!("solve failed: {err}");
                std::process::exit(1);
            }
            info!("command solve completed");
        }
        Some(Commands::Search {
            query,
            repo,
            with_scores,
            refresh_index,
        }) => {
            info!(repo = %repo.display(), with_scores, refresh_index, "command search start");
            if let Err(err) = run_search(config, repo, &query, with_scores, refresh_index).await {
                warn!(error = %err, "command search failed");
                eprintln!("search failed: {err}");
                std::process::exit(1);
            }
            info!("command search completed");
        }
        None => {
            println!("Use `warlock solve|index|plan|search --help`");
        }
    }
}

async fn run_plan_only(
    config: Config,
    repo: PathBuf,
    goal: &str,
    verbose_planning: bool,
) -> anyhow::Result<()> {
    info!(repo = %repo.display(), goal_len = goal.len(), "plan-only pipeline start");
    if verbose_planning {
        if let Err(err) = print_verbose_planning_context(&config, &repo, goal).await {
            warn!(error = %err, "verbose planning diagnostics failed");
            eprintln!("warning: verbose planning diagnostics failed: {err}");
        }
    }

    info!("plan-only: creating llm provider");
    let llm = build_provider(&config.llm.provider)?;
    let completion = CompletionConfig {
        model: config.llm.model.clone(),
        temperature: config.llm.temperature,
        max_tokens: config.llm.max_tokens,
        json_mode: true,
    };

    let planner = Planner::new(llm, completion);
    info!("plan-only: indexing repo for summary context");
    let chunks = index_repo_async(&repo, &config.indexing).await?;
    debug!(chunks = chunks.len(), "plan-only: indexed chunks");
    let summary = build_planning_summary(&chunks);
    let plan = planner
        .generate_plan_with_limit(goal, &summary, Some(config.agent.planner_max_steps))
        .await?;
    info!(steps = plan.steps.len(), "plan-only: plan generated");
    if let Err(err) = save_plan_markdown(&repo, goal, &plan) {
        warn!(error = %err, "failed to persist plan markdown");
        eprintln!("warning: failed to persist plan markdown: {err}");
    }
    println!("{}", serde_json::to_string_pretty(&plan)?);
    Ok(())
}

async fn run_solve(
    config: Config,
    repo: PathBuf,
    goal: &str,
    verbose_planning: bool,
) -> anyhow::Result<()> {
    info!(repo = %repo.display(), goal_len = goal.len(), "solve pipeline start");
    if verbose_planning {
        if let Err(err) = print_verbose_planning_context(&config, &repo, goal).await {
            warn!(error = %err, "verbose planning diagnostics failed");
            eprintln!("warning: verbose planning diagnostics failed: {err}");
        }
    }

    info!("solve: creating llm provider and agents");
    let llm = build_provider(&config.llm.provider)?;
    let planner = Planner::new(
        llm.clone(),
        CompletionConfig {
            model: config.llm.model.clone(),
            temperature: config.llm.temperature,
            max_tokens: config.llm.max_tokens,
            json_mode: true,
        },
    );
    let coder = Coder::new(
        llm.clone(),
        CompletionConfig {
            model: config.llm.model.clone(),
            temperature: config.llm.temperature,
            max_tokens: config.llm.max_tokens,
            json_mode: false,
        },
    );
    let reflector = Reflector::new(
        llm,
        CompletionConfig {
            model: config.llm.model.clone(),
            temperature: config.llm.temperature,
            max_tokens: config.llm.max_tokens,
            json_mode: false,
        },
    );

    let mut orchestrator =
        Orchestrator::new(config, repo.clone(), planner, coder, reflector).await?;
    info!("solve: orchestrator started");
    let report = orchestrator.solve(goal).await?;
    info!(
        success = report.overall_success,
        duration_secs = report.duration_secs,
        "solve finished"
    );
    if let Err(err) = save_report_markdown(&repo, goal, &report) {
        warn!(error = %err, "failed to persist run report markdown");
        eprintln!("warning: failed to persist run report markdown: {err}");
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

async fn run_search(
    config: Config,
    repo: PathBuf,
    query: &str,
    with_scores: bool,
    refresh_index: bool,
) -> anyhow::Result<()> {
    use anyhow::Context;

    info!(repo = %repo.display(), query_len = query.len(), with_scores, refresh_index, "search pipeline start");
    let repo_id = derive_repo_id(&repo);
    let embedder = Embedder::new(&config.embeddings)?;
    let embedding_cache = if config.embeddings.persist_cache {
        let db_path = config
            .embeddings
            .cache_db_path
            .as_ref()
            .map(PathBuf::from)
            .map(|p| if p.is_absolute() { p } else { repo.join(p) })
            .unwrap_or_else(|| repo.join(".warlock/embeddings_cache"));
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

    sync_vector_index_with_components(
        &config,
        &repo,
        refresh_index,
        &repo_id,
        &embedder,
        &embedding_cache,
        &store,
    )
    .await?;
    info!(repo_id = %repo_id, "search: vector index synchronized");

    let query_vec = embedder
        .embed_batch(&[query.to_string()])
        .await?
        .into_iter()
        .next()
        .context("failed to embed search query")?;

    let rewritten_queries = rewrite_query(query);
    debug!(
        rewrites = rewritten_queries.len(),
        "search: query rewrites computed"
    );
    let rewritten_query_vecs = if rewritten_queries.is_empty() {
        vec![query_vec.clone()]
    } else {
        embedder.embed_batch_chunked(&rewritten_queries).await?
    };

    let mut aggregated_hits = Vec::new();
    for vec in rewritten_query_vecs {
        let hits = store
            .search_with_scores(
                &repo_id,
                vec,
                (config.vector_store.top_k as usize) * 2,
                None,
            )
            .await?;
        aggregated_hits.extend(hits);
    }

    let ranked_hits = hybrid_rank_hits(query, aggregated_hits, 0.7);
    info!(hits = ranked_hits.len(), "search: ranking complete");

    if with_scores {
        let hits = ranked_hits
            .into_iter()
            .take(config.vector_store.top_k as usize)
            .collect::<Vec<_>>();
        println!("{}", serde_json::to_string_pretty(&hits)?);
    } else {
        let chunks = ranked_hits
            .into_iter()
            .take(config.vector_store.top_k as usize)
            .map(|h| h.chunk)
            .collect::<Vec<_>>();
        println!("{}", serde_json::to_string_pretty(&chunks)?);
    }

    Ok(())
}

async fn sync_vector_index(
    config: &Config,
    repo: &std::path::Path,
    force_refresh: bool,
) -> anyhow::Result<()> {
    info!(repo = %repo.display(), force_refresh, "sync_vector_index start");
    let repo_id = derive_repo_id(repo);
    let embedder = Embedder::new(&config.embeddings)?;
    let embedding_cache = if config.embeddings.persist_cache {
        let db_path = config
            .embeddings
            .cache_db_path
            .as_ref()
            .map(PathBuf::from)
            .map(|p| if p.is_absolute() { p } else { repo.join(p) })
            .unwrap_or_else(|| repo.join(".warlock/embeddings_cache"));
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

    sync_vector_index_with_components(
        config,
        repo,
        force_refresh,
        &repo_id,
        &embedder,
        &embedding_cache,
        &store,
    )
    .await
}

async fn sync_vector_index_with_components(
    config: &Config,
    repo: &std::path::Path,
    force_refresh: bool,
    repo_id: &str,
    embedder: &Embedder,
    embedding_cache: &EmbeddingCache,
    store: &VectorStore,
) -> anyhow::Result<()> {
    info!(repo = %repo.display(), repo_id = %repo_id, force_refresh, "index sync start");
    let manifest_path = RepositoryManifest::default_path(repo);
    let current_manifest = RepositoryManifest::build(
        repo,
        &config.indexing,
        repo_id,
        &config.embeddings.model,
        config.embeddings.dimensions,
    )?;
    let stored_manifest = RepositoryManifest::load(&manifest_path)?;
    let diff = current_manifest.diff_from(stored_manifest.as_ref());

    let has_chunks = store.has_code_chunks(repo_id).await?;
    let has_file_summaries = store
        .has_doc_type(repo_id, VectorDocType::FileSummary)
        .await?;
    let has_symbol_summaries = store
        .has_doc_type(repo_id, VectorDocType::SymbolSummary)
        .await?;

    let force_full = force_refresh
        || diff.requires_full_reindex
        || !has_chunks
        || !has_file_summaries
        || !has_symbol_summaries;

    info!(
        repo_id = %repo_id,
        changed = diff.changed_or_new.len(),
        removed = diff.removed.len(),
        requires_full_reindex = diff.requires_full_reindex,
        has_chunks,
        has_file_summaries,
        has_symbol_summaries,
        force_full,
        "index sync decision"
    );

    if force_full {
        info!(repo_id = %repo_id, "index sync full reindex start");
        eprintln!("indexing repository chunks + summaries into vector store...");
        tokio::try_join!(
            store.delete_doc_type(repo_id, VectorDocType::CodeChunk),
            store.delete_doc_type(repo_id, VectorDocType::FileSummary),
            store.delete_doc_type(repo_id, VectorDocType::SymbolSummary),
        )?;

        let chunks = index_repo_async(repo, &config.indexing).await?;
        if !chunks.is_empty() {
            let chunk_embeddings = embedder
                .embed_chunks_with_cache(&chunks, embedding_cache)
                .await?;
            store.upsert(repo_id, &chunks, &chunk_embeddings).await?;
            upsert_summaries(repo_id, &chunks, embedder, store).await?;
        }

        current_manifest.save(&manifest_path)?;
        info!(repo_id = %repo_id, "index sync full reindex completed");
        return Ok(());
    }

    if diff.changed_or_new.is_empty() && diff.removed.is_empty() {
        info!(repo_id = %repo_id, "index sync no-op; no changes detected");
        return Ok(());
    }

    info!(repo_id = %repo_id, changed = diff.changed_or_new.len(), removed = diff.removed.len(), "index sync incremental start");

    for removed in &diff.removed {
        store.delete_by_file(repo_id, removed).await?;
    }

    if !diff.changed_or_new.is_empty() {
        let changed_abs = diff
            .changed_or_new
            .iter()
            .map(|rel| repo.join(rel))
            .filter(|path| path.exists())
            .collect::<Vec<_>>();
        debug!(repo_id = %repo_id, existing_changed_files = changed_abs.len(), "index sync incremental file set");

        for rel in &diff.changed_or_new {
            store.delete_by_file(repo_id, rel).await?;
        }

        let chunks = index_files_async(repo, changed_abs, &config.indexing).await?;
        if !chunks.is_empty() {
            let chunk_embeddings = embedder
                .embed_chunks_with_cache(&chunks, embedding_cache)
                .await?;
            store.upsert(repo_id, &chunks, &chunk_embeddings).await?;
            upsert_summaries(repo_id, &chunks, embedder, store).await?;
        }
    }

    current_manifest.save(&manifest_path)?;
    info!(repo_id = %repo_id, "index sync incremental completed");
    Ok(())
}

async fn upsert_summaries(
    repo_id: &str,
    chunks: &[CodeChunk],
    embedder: &Embedder,
    store: &VectorStore,
) -> anyhow::Result<()> {
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
        let embeddings = embedder.embed_batch_chunked(&texts).await?;
        store
            .upsert_file_summaries(repo_id, &file_summaries, &embeddings)
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
        let embeddings = embedder.embed_batch_chunked(&texts).await?;
        store
            .upsert_symbol_summaries(repo_id, &symbol_summaries, &embeddings)
            .await
    };

    tokio::try_join!(file_task, symbol_task)?;
    Ok(())
}

fn save_plan_markdown(repo: &std::path::Path, goal: &str, plan: &Plan) -> anyhow::Result<PathBuf> {
    let timestamp = unix_timestamp_millis();
    let path = repo
        .join(".warlock")
        .join("plans")
        .join(format!("{timestamp}_plan.md"));

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut md = String::new();
    md.push_str("# Warlock Plan\n\n");
    md.push_str(&format!("- Timestamp: `{timestamp}`\n"));
    md.push_str(&format!("- Goal: {}\n", goal.trim()));
    md.push_str(&format!("- Steps: {}\n\n", plan.steps.len()));

    md.push_str("## Plan Checklist\n\n");
    for step in &plan.steps {
        let targets = if step.target_files.is_empty() {
            "(none)".to_string()
        } else {
            step.target_files
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let deps = if step.depends_on.is_empty() {
            "none".to_string()
        } else {
            step.depends_on
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };

        md.push_str(&format!("- [ ] Step {}: {}\n", step.step, step.task));
        md.push_str(&format!("  - Targets: {}\n", targets));
        md.push_str(&format!("  - Depends on: {}\n", deps));
        if let Some(hypothesis) = &step.hypothesis {
            md.push_str(&format!("  - Hypothesis: {}\n", hypothesis.trim()));
        }
        if !step.predicted_consequences.is_empty() {
            md.push_str("  - Predicted consequences:\n");
            for consequence in &step.predicted_consequences {
                md.push_str(&format!("    - {}\n", consequence.trim()));
            }
        }
        if let Some(experiment) = &step.experiment {
            md.push_str(&format!(
                "  - Experiment command: `{}`\n",
                experiment.command.trim()
            ));
            if !experiment.success_contains.is_empty() {
                md.push_str("  - Experiment success checks:\n");
                for check in &experiment.success_contains {
                    md.push_str(&format!("    - {}\n", check.trim()));
                }
            }
        }
        if let Some(confidence) = step.confidence {
            md.push_str(&format!("  - Confidence: {:.2}\n", confidence));
        }
    }

    std::fs::write(&path, md)?;
    Ok(path)
}

async fn print_verbose_planning_context(
    config: &Config,
    repo: &std::path::Path,
    goal: &str,
) -> anyhow::Result<()> {
    use anyhow::Context;

    let repo_id = derive_repo_id(repo);
    let store = VectorStore::new(
        &config.vector_store.url,
        &config.vector_store.collection,
        config.embeddings.dimensions as u64,
    )
    .await?;
    let embedder = Embedder::new(&config.embeddings)?;

    let query_vec = embedder
        .embed_batch(&[goal.to_string()])
        .await?
        .into_iter()
        .next()
        .context("failed to embed planning goal")?;

    let top_k = std::cmp::max(1, std::cmp::min(config.vector_store.top_k as usize, 8));
    let chunk_hits = store
        .search_with_scores(&repo_id, query_vec.clone(), top_k, None)
        .await?;
    let symbol_hits = store
        .search_summaries_with_scores(
            &repo_id,
            query_vec.clone(),
            top_k,
            VectorDocType::SymbolSummary,
        )
        .await?;
    let file_hits = store
        .search_summaries_with_scores(&repo_id, query_vec, top_k, VectorDocType::FileSummary)
        .await?;

    let has_chunks = store.has_code_chunks(&repo_id).await?;
    let has_symbols = store
        .has_doc_type(&repo_id, VectorDocType::SymbolSummary)
        .await?;
    let has_files = store
        .has_doc_type(&repo_id, VectorDocType::FileSummary)
        .await?;

    eprintln!("planning.verbose: enabled=true");
    eprintln!(
        "planning.verbose: qdrant.url={}, collection={}",
        config.vector_store.url, config.vector_store.collection
    );
    eprintln!("planning.verbose: repo_id={repo_id}");
    eprintln!(
        "planning.verbose: qdrant.queried=true, top_k={}, goal='{}'",
        top_k,
        goal.trim()
    );
    eprintln!(
        "planning.verbose: qdrant.presence code_chunks={}, symbol_summaries={}, file_summaries={}",
        has_chunks, has_symbols, has_files
    );
    eprintln!(
        "planning.verbose: qdrant.hits code_chunks={}, symbol_summaries={}, file_summaries={}",
        chunk_hits.len(),
        symbol_hits.len(),
        file_hits.len()
    );

    Ok(())
}

fn save_report_markdown(
    repo: &std::path::Path,
    goal: &str,
    report: &SolveReport,
) -> anyhow::Result<PathBuf> {
    let timestamp = unix_timestamp_millis();
    let path = repo
        .join(".warlock")
        .join("runs")
        .join(format!("{timestamp}_report.md"));

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut md = String::new();
    md.push_str("# Warlock Solve Report\n\n");
    md.push_str(&format!("- Timestamp: `{timestamp}`\n"));
    md.push_str(&format!("- Goal: {}\n", goal.trim()));
    md.push_str(&format!(
        "- Overall success: {}\n",
        if report.overall_success { "yes" } else { "no" }
    ));
    md.push_str(&format!(
        "- Duration (seconds): {:.3}\n\n",
        report.duration_secs
    ));

    md.push_str("## Plan Progress\n\n");
    if let Some(plan) = &report.plan {
        let latest = latest_step_statuses(&report.step_results);
        for step in &plan.steps {
            let marker = checkbox_for_step(step.step, &latest);
            let note = status_note_for_step(step.step, &latest);
            md.push_str(&format!(
                "- {} Step {}: {}{}\n",
                marker, step.step, step.task, note
            ));
        }
    } else {
        md.push_str("- No plan was produced.\n");
    }

    md.push_str("\n## Execution Details\n\n");
    if report.step_results.is_empty() {
        md.push_str("No execution steps were recorded.\n");
    } else {
        for (idx, step) in report.step_results.iter().enumerate() {
            md.push_str(&format!(
                "{}. Step {} — **{}**\n",
                idx + 1,
                step.step,
                step.task
            ));
            md.push_str(&format!("   - Status: {}\n", display_status(&step.status)));
            md.push_str(&format!("   - Attempts: {}\n", step.attempts));
            if let Some(experiment) = &step.experiment {
                md.push_str(&format!(
                    "   - Hypothesis: {}\n",
                    experiment.hypothesis.replace('\n', " ")
                ));
                md.push_str(&format!(
                    "   - Experiment command: `{}`\n",
                    experiment.command
                ));
                md.push_str(&format!(
                    "   - Experiment passed: {}\n",
                    if experiment.passed { "yes" } else { "no" }
                ));
                if !experiment.notes.trim().is_empty() {
                    md.push_str(&format!(
                        "   - Experiment notes: {}\n",
                        experiment.notes.replace('\n', " ")
                    ));
                }
            }
            if let Some(err) = &step.error {
                md.push_str(&format!("   - Error: {}\n", err.replace('\n', " ")));
            }
            md.push('\n');
        }
    }

    std::fs::write(&path, md)?;
    Ok(path)
}

fn latest_step_statuses(step_results: &[StepResult]) -> HashMap<u32, StepResult> {
    let mut latest = HashMap::new();
    for result in step_results {
        latest.insert(result.step, result.clone());
    }
    latest
}

fn checkbox_for_step(step: u32, latest: &HashMap<u32, StepResult>) -> &'static str {
    match latest.get(&step).map(|s| &s.status) {
        Some(StepStatus::Succeeded) => "[x]",
        _ => "[ ]",
    }
}

fn status_note_for_step(step: u32, latest: &HashMap<u32, StepResult>) -> String {
    match latest.get(&step) {
        Some(result) => match result.status {
            StepStatus::Succeeded => " ✅ completed".to_string(),
            StepStatus::Failed if result.attempts > 0 => {
                format!(" ⚠️ partial (attempted {}, failed)", result.attempts)
            }
            StepStatus::Failed => " ❌ failed".to_string(),
            StepStatus::Skipped => " ⏭️ not done (skipped)".to_string(),
        },
        None => " ⏳ not started".to_string(),
    }
}

fn display_status(status: &StepStatus) -> &'static str {
    match status {
        StepStatus::Succeeded => "succeeded",
        StepStatus::Failed => "failed",
        StepStatus::Skipped => "skipped",
    }
}

fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}
