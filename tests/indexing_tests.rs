use std::path::Path;
use warlock::{
    config::Config,
    indexing::{
        build_file_summaries, build_module_summaries, build_planning_summary,
        build_symbol_summaries, index_repo, index_repo_async, CodeChunk, SymbolKind,
    },
};

#[test]
fn indexes_mixed_language_symbols() {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("warlock.toml");
    let config = Config::load(config_path).expect("load config");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/indexing");

    let chunks = index_repo(&repo, &config.indexing).expect("indexing should succeed");
    assert!(!chunks.is_empty());

    let symbols: Vec<&str> = chunks.iter().map(|c| c.symbol.as_str()).collect();
    assert!(symbols.contains(&"add"));
    assert!(symbols.contains(&"Greeter"));
    assert!(symbols.contains(&"multiply"));
    assert!(symbols.contains(&"Counter"));
    assert!(symbols.contains(&"sum"));
}

#[test]
fn skips_generated_files_and_splits_large_chunks() {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("warlock.toml");
    let mut config = Config::load(config_path).expect("load config");
    config.indexing.max_chunk_lines = 8;

    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/indexing");
    let chunks = index_repo(&repo, &config.indexing).expect("indexing should succeed");

    assert!(
        chunks
            .iter()
            .all(|c| !c.file.to_string_lossy().contains("generated.rs")),
        "generated file should be skipped"
    );

    let long_fn_parts = chunks
        .iter()
        .filter(|c| c.symbol.starts_with("long_calc#part"))
        .count();
    assert!(
        long_fn_parts >= 2,
        "large symbol should be split into multiple chunks"
    );
}

#[test]
fn indexing_order_is_stable() {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("warlock.toml");
    let config = Config::load(config_path).expect("load config");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/indexing");

    let first = index_repo(&repo, &config.indexing).expect("first run");
    let second = index_repo(&repo, &config.indexing).expect("second run");

    let first_keys: Vec<_> = first
        .iter()
        .map(|c| (c.file.clone(), c.symbol.clone(), c.span))
        .collect();
    let second_keys: Vec<_> = second
        .iter()
        .map(|c| (c.file.clone(), c.symbol.clone(), c.span))
        .collect();

    assert_eq!(first_keys, second_keys);
}

#[tokio::test(flavor = "multi_thread")]
async fn async_indexing_matches_sync_output() {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("warlock.toml");
    let mut config = Config::load(config_path).expect("load config");
    config.indexing.parallel_file_workers = 4;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/indexing");

    let sync = index_repo(&repo, &config.indexing).expect("sync indexing should succeed");
    let async_chunks = index_repo_async(&repo, &config.indexing)
        .await
        .expect("async indexing should succeed");

    let sync_keys: Vec<_> = sync
        .iter()
        .map(|c| (c.file.clone(), c.symbol.clone(), c.span))
        .collect();
    let async_keys: Vec<_> = async_chunks
        .iter()
        .map(|c| (c.file.clone(), c.symbol.clone(), c.span))
        .collect();

    assert_eq!(sync_keys, async_keys);
}

#[test]
fn builds_file_and_symbol_summaries() {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("warlock.toml");
    let config = Config::load(config_path).expect("load config");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/indexing");

    let chunks = index_repo(&repo, &config.indexing).expect("indexing should succeed");
    let file_summaries = build_file_summaries(&chunks);
    let symbol_summaries = build_symbol_summaries(&chunks);

    assert!(!file_summaries.is_empty());
    assert!(!symbol_summaries.is_empty());
    assert!(symbol_summaries
        .iter()
        .any(|s| s.symbol == "add" || s.symbol == "multiply"));
    assert!(
        symbol_summaries
            .iter()
            .all(|s| !s.summary.contains("Signature/context:")),
        "symbol summaries should describe behavior rather than raw signature"
    );

    let sum_summary = symbol_summaries
        .iter()
        .find(|s| s.symbol == "sum")
        .expect("sum symbol summary should exist");
    assert!(
        sum_summary.summary.contains("aggregates")
            || sum_summary.summary.contains("fold")
            || sum_summary.summary.contains("computes"),
        "sum summary should include behavior-oriented wording"
    );
}

#[test]
fn planning_summary_filters_unknown_fallback_entries() {
    let chunks = vec![
        CodeChunk {
            file: "src/agents/planner.rs".into(),
            symbol: "generate_plan_with_limit".to_string(),
            kind: SymbolKind::Function,
            code: "pub fn generate_plan_with_limit() {}".to_string(),
            span: (0, 10),
            ast_sexp: "(function_item)".to_string(),
        },
        CodeChunk {
            file: "src/agents/planner.rs".into(),
            symbol: "src/agents/planner.rs#fallback1".to_string(),
            kind: SymbolKind::Unknown,
            code: "// fallback".to_string(),
            span: (10, 20),
            ast_sexp: "(fallback_line_chunk)".to_string(),
        },
    ];

    let summary = build_planning_summary(&chunks);
    assert!(summary.contains("Module summaries:"));
    assert!(summary.contains("agents::planner"));
    assert!(summary.contains("Unknown fallback chunks: 1"));
    assert!(!summary.contains("::Unknown::"));
    assert!(!summary.contains("#fallback"));
}

#[test]
fn module_summaries_group_by_source_module_path() {
    let chunks = vec![
        CodeChunk {
            file: "src/agents/planner.rs".into(),
            symbol: "generate_plan".to_string(),
            kind: SymbolKind::Function,
            code: "pub fn generate_plan() {}".to_string(),
            span: (0, 10),
            ast_sexp: "(function_item)".to_string(),
        },
        CodeChunk {
            file: "src/agents/reflector.rs".into(),
            symbol: "reflect".to_string(),
            kind: SymbolKind::Function,
            code: "pub fn reflect() {}".to_string(),
            span: (0, 10),
            ast_sexp: "(function_item)".to_string(),
        },
    ];

    let symbol_summaries = build_symbol_summaries(&chunks);
    let module_summaries = build_module_summaries(&symbol_summaries);
    let modules = module_summaries
        .iter()
        .map(|m| m.module.clone())
        .collect::<Vec<_>>();
    assert!(modules.contains(&"agents::planner".to_string()));
    assert!(modules.contains(&"agents::reflector".to_string()));
}
