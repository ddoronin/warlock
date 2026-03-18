pub mod chunker;
pub mod manifest;
pub mod parser;
pub mod walker;

use crate::config::IndexingConfig;
use anyhow::{Context, Result};
use futures::{stream, StreamExt, TryStreamExt};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use chunker::{CodeChunk, SymbolKind};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct FileSummary {
    pub file: std::path::PathBuf,
    pub summary: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SymbolSummary {
    pub file: std::path::PathBuf,
    pub symbol: String,
    pub kind: SymbolKind,
    pub summary: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ModuleSummary {
    pub module: String,
    pub summary: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct FolderSummary {
    pub folder: String,
    pub summary: String,
}

/// Indexes a repository into semantic code chunks.
pub fn index_repo(repo_root: impl AsRef<Path>, config: &IndexingConfig) -> Result<Vec<CodeChunk>> {
    let repo_root = repo_root.as_ref();
    let files = walker::discover_files(repo_root, config)?;
    index_files(repo_root, files, config)
}

/// Indexes a repository into semantic code chunks using async parallel file processing.
pub async fn index_repo_async(
    repo_root: impl AsRef<Path>,
    config: &IndexingConfig,
) -> Result<Vec<CodeChunk>> {
    let repo_root = repo_root.as_ref();
    let files = walker::discover_files(repo_root, config)?;
    index_files_async(repo_root, files, config).await
}

pub fn index_files(
    repo_root: &Path,
    files: Vec<std::path::PathBuf>,
    config: &IndexingConfig,
) -> Result<Vec<CodeChunk>> {
    let mut chunks = Vec::new();
    let mut parsers = parser::LanguageParsers::new()?;

    for file in files {
        chunks.extend(index_single_file(
            repo_root,
            &file,
            &config.supported_languages,
            config.max_chunk_lines,
            &mut parsers,
        )?);
    }

    sort_chunks(&mut chunks);
    Ok(chunks)
}

pub async fn index_files_async(
    repo_root: &Path,
    files: Vec<std::path::PathBuf>,
    config: &IndexingConfig,
) -> Result<Vec<CodeChunk>> {
    if files.is_empty() {
        return Ok(Vec::new());
    }

    let repo_root = repo_root.to_path_buf();
    let supported_languages = Arc::new(config.supported_languages.clone());
    let max_chunk_lines = config.max_chunk_lines;
    let worker_limit = config.parallel_file_workers.max(1);

    let mut per_file_chunks = stream::iter(files.into_iter().enumerate())
        .map(|(idx, file)| {
            let repo_root = repo_root.clone();
            let supported_languages = supported_languages.clone();

            async move {
                let output = tokio::task::spawn_blocking(move || {
                    let mut parsers = parser::LanguageParsers::new()?;
                    let file_chunks = index_single_file(
                        &repo_root,
                        &file,
                        &supported_languages,
                        max_chunk_lines,
                        &mut parsers,
                    )?;
                    Ok::<(usize, Vec<CodeChunk>), anyhow::Error>((idx, file_chunks))
                })
                .await
                .context("indexing worker panicked")??;

                Ok::<(usize, Vec<CodeChunk>), anyhow::Error>(output)
            }
        })
        .buffer_unordered(worker_limit)
        .try_collect::<Vec<_>>()
        .await?;

    per_file_chunks.sort_by_key(|(idx, _)| *idx);

    let mut chunks = Vec::new();
    for (_, mut file_chunks) in per_file_chunks {
        chunks.append(&mut file_chunks);
    }

    sort_chunks(&mut chunks);

    Ok(chunks)
}

fn index_single_file(
    repo_root: &Path,
    file: &Path,
    supported_languages: &[String],
    max_chunk_lines: usize,
    parsers: &mut parser::LanguageParsers,
) -> Result<Vec<CodeChunk>> {
    let parse = parser::parse_file(repo_root, file, parsers, supported_languages)?;
    if let Some(parsed) = parse {
        Ok(chunker::extract_chunks(&parsed, max_chunk_lines))
    } else {
        Ok(Vec::new())
    }
}

fn sort_chunks(chunks: &mut [CodeChunk]) {
    chunks.sort_by(|a, b| {
        a.file
            .cmp(&b.file)
            .then_with(|| a.span.0.cmp(&b.span.0))
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
}

pub fn build_planning_summary(chunks: &[CodeChunk]) -> String {
    const MAX_MODULE_LINES: usize = 24;
    const MAX_SYMBOL_LINES: usize = 140;

    let mut files = BTreeSet::new();
    let mut unknown_chunks = 0usize;
    for chunk in chunks {
        files.insert(chunk.file.clone());
        if chunk.kind == SymbolKind::Unknown {
            unknown_chunks += 1;
        }
    }

    let symbol_summaries = build_symbol_summaries(chunks);
    let module_summaries = build_module_summaries(&symbol_summaries);

    let mut kind_counts: BTreeMap<String, usize> = BTreeMap::new();
    for symbol in &symbol_summaries {
        *kind_counts
            .entry(format!("{:?}", symbol.kind).to_ascii_lowercase())
            .or_default() += 1;
    }

    let symbol_kind_lines = if kind_counts.is_empty() {
        "- (none)".to_string()
    } else {
        kind_counts
            .into_iter()
            .map(|(kind, count)| format!("- {kind}: {count}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let module_lines = if module_summaries.is_empty() {
        "- (none)".to_string()
    } else {
        module_summaries
            .into_iter()
            .take(MAX_MODULE_LINES)
            .map(|m| format!("- {}", m.summary))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let mut representative_symbols = symbol_summaries
        .iter()
        .map(|s| {
            format!(
                "{}::{} ({})",
                derive_module_name(&s.file),
                s.symbol,
                format!("{:?}", s.kind).to_ascii_lowercase()
            )
        })
        .collect::<Vec<_>>();
    representative_symbols.sort();
    representative_symbols.dedup();

    let symbol_lines = if representative_symbols.is_empty() {
        "- (none)".to_string()
    } else {
        representative_symbols
            .into_iter()
            .take(MAX_SYMBOL_LINES)
            .map(|s| format!("- {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "Files: {}\nChunks: {}\nKnown symbols: {}\nUnknown fallback chunks: {}\n\nSymbol kinds:\n{}\n\nModule summaries:\n{}\n\nRepresentative symbols:\n{}",
        files.len(),
        chunks.len(),
        symbol_summaries.len(),
        unknown_chunks,
        symbol_kind_lines,
        module_lines,
        symbol_lines,
    )
}

pub fn build_file_summaries(chunks: &[CodeChunk]) -> Vec<FileSummary> {
    let mut by_file: BTreeMap<&std::path::PathBuf, Vec<&CodeChunk>> = BTreeMap::new();
    for chunk in chunks {
        by_file.entry(&chunk.file).or_default().push(chunk);
    }

    let mut summaries = Vec::with_capacity(by_file.len());
    for (file, file_chunks) in by_file {
        let mut kinds = BTreeSet::new();
        let mut symbols = Vec::new();
        for chunk in &file_chunks {
            if chunk.kind != SymbolKind::Unknown {
                kinds.insert(format!("{:?}", chunk.kind).to_ascii_lowercase());
            }
            if symbols.len() < 5 {
                symbols.push(normalize_symbol_name(&chunk.symbol).to_string());
            }
        }

        let behavior = infer_behavior_from_text(
            &file_chunks
                .iter()
                .map(|c| c.code.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let summary = format!(
            "File {} defines {} chunk(s), symbol kinds [{}], key symbols [{}]. Primary behavior: {}.",
            file.display(),
            file_chunks.len(),
            kinds.into_iter().collect::<Vec<_>>().join(", "),
            symbols.join(", "),
            behavior
        );

        summaries.push(FileSummary {
            file: file.clone(),
            summary,
        });
    }

    summaries
}

pub fn build_symbol_summaries(chunks: &[CodeChunk]) -> Vec<SymbolSummary> {
    let mut seen = BTreeSet::new();
    let mut summaries = Vec::new();

    for chunk in chunks {
        if chunk.kind == SymbolKind::Unknown {
            continue;
        }
        let base_symbol = normalize_symbol_name(&chunk.symbol).to_string();
        let key = format!(
            "{}:{}:{}",
            chunk.file.display(),
            base_symbol,
            format!("{:?}", chunk.kind)
        );
        if !seen.insert(key) {
            continue;
        }

        let kind_label = symbol_kind_label(&chunk.kind);
        let behavior = infer_symbol_behavior(&base_symbol, &chunk.code);
        let summary = if let Some(doc) = extract_inline_doc_summary(&chunk.code) {
            format!(
                "{} `{}` in {}. {} Behavior: {}.",
                kind_label,
                base_symbol,
                chunk.file.display(),
                doc,
                behavior
            )
        } else {
            format!(
                "{} `{}` in {} {}.",
                kind_label,
                base_symbol,
                chunk.file.display(),
                behavior
            )
        };

        summaries.push(SymbolSummary {
            file: chunk.file.clone(),
            symbol: base_symbol,
            kind: chunk.kind.clone(),
            summary,
        });
    }

    summaries
}

fn normalize_symbol_name(symbol: &str) -> &str {
    symbol.split('#').next().unwrap_or(symbol)
}

pub fn build_module_summaries(symbol_summaries: &[SymbolSummary]) -> Vec<ModuleSummary> {
    let mut grouped: BTreeMap<String, Vec<&SymbolSummary>> = BTreeMap::new();
    for summary in symbol_summaries {
        let module = derive_module_name(&summary.file);
        grouped.entry(module).or_default().push(summary);
    }

    grouped
        .into_iter()
        .map(|(module, items)| ModuleSummary {
            module: module.clone(),
            summary: format!(
                "Module {} has {} symbol(s). Representative behaviors: {}.",
                module,
                items.len(),
                items
                    .iter()
                    .take(5)
                    .map(|s| format!("{}: {}", s.symbol, compact_summary(&s.summary)))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
        })
        .collect()
}

fn derive_module_name(file: &PathBuf) -> String {
    let mut parts = file
        .with_extension("")
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect::<Vec<_>>();

    if parts.first().is_some_and(|p| p == "src") {
        parts.remove(0);
    }

    if parts.is_empty() {
        "root".to_string()
    } else {
        parts.join("::")
    }
}

pub fn build_folder_summaries(file_summaries: &[FileSummary]) -> Vec<FolderSummary> {
    let mut grouped: BTreeMap<String, Vec<&FileSummary>> = BTreeMap::new();
    for summary in file_summaries {
        let folder = summary
            .file
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        grouped.entry(folder).or_default().push(summary);
    }

    grouped
        .into_iter()
        .map(|(folder, items)| FolderSummary {
            summary: format!(
                "Folder `{}` groups {} file(s). Dominant responsibilities: {}.",
                folder,
                items.len(),
                items
                    .iter()
                    .take(4)
                    .map(|s| compact_summary(&s.summary))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            folder,
        })
        .collect()
}

fn compact_summary(summary: &str) -> String {
    summary
        .split('.')
        .next()
        .unwrap_or(summary)
        .trim()
        .to_string()
}

fn symbol_kind_label(kind: &SymbolKind) -> &'static str {
    match kind {
        SymbolKind::Function => "Function",
        SymbolKind::Struct => "Struct",
        SymbolKind::Enum => "Enum",
        SymbolKind::Trait => "Trait",
        SymbolKind::Impl => "Impl block",
        SymbolKind::Module => "Module",
        SymbolKind::Class => "Class",
        SymbolKind::Method => "Method",
        SymbolKind::Unknown => "Symbol",
    }
}

fn extract_inline_doc_summary(code: &str) -> Option<String> {
    let mut docs = Vec::new();
    for line in code.lines().take(12) {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("///") {
            docs.push(rest.trim().to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("//") {
            if !rest.trim().is_empty() {
                docs.push(rest.trim().to_string());
            }
            continue;
        }
        if trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("struct ")
        {
            break;
        }
    }

    if docs.is_empty() {
        None
    } else {
        Some(docs.join(" "))
    }
}

fn infer_symbol_behavior(symbol_name: &str, code: &str) -> String {
    let mut actions = infer_behavior_from_name(symbol_name);
    actions.extend(infer_actions_from_text(code));
    actions.sort();
    actions.dedup();

    if actions.is_empty() {
        "computes derived output from inputs".to_string()
    } else {
        actions.join(", ")
    }
}

fn infer_behavior_from_text(code: &str) -> String {
    let mut actions = infer_actions_from_text(code);
    actions.sort();
    actions.dedup();

    if actions.is_empty() {
        "implements internal project logic".to_string()
    } else {
        actions.join(", ")
    }
}

fn infer_behavior_from_name(symbol_name: &str) -> Vec<String> {
    let name = symbol_name.to_ascii_lowercase();
    let mut actions = Vec::new();

    if name.contains("sum") || name.contains("total") || name.contains("average") {
        actions.push("aggregates numeric values".to_string());
    }
    if name.contains("count") {
        actions.push("counts matching entries".to_string());
    }
    if name.starts_with("get") || name.starts_with("find") || name.starts_with("load") {
        actions.push("retrieves matching data".to_string());
    }
    if name.starts_with("set") || name.starts_with("update") || name.starts_with("apply") {
        actions.push("updates state or applies changes".to_string());
    }
    if name.starts_with("parse") || name.starts_with("decode") {
        actions.push("parses structured input".to_string());
    }
    if name.starts_with("validate") || name.starts_with("check") {
        actions.push("validates constraints".to_string());
    }

    actions
}

fn infer_actions_from_text(code: &str) -> Vec<String> {
    let lower = code.to_ascii_lowercase();
    let mut actions = Vec::new();

    if ["write(", "insert", "update", "delete", "remove", "push("]
        .iter()
        .any(|k| lower.contains(k))
    {
        actions.push("mutates state".to_string());
    }

    if [
        "read_to_string",
        "std::fs",
        "open(",
        "create_dir",
        "write_all",
    ]
    .iter()
    .any(|k| lower.contains(k))
    {
        actions.push("performs filesystem i/o".to_string());
    }

    if ["reqwest", "http", "client.", "post_json", "put_json"]
        .iter()
        .any(|k| lower.contains(k))
    {
        actions.push("calls external services".to_string());
    }

    if ["reduce(", ".sum(", "+", "acc", "total"]
        .iter()
        .any(|k| lower.contains(k))
    {
        actions.push("aggregates or folds collections".to_string());
    }

    if ["serde_json", "json!", "toml::", "from_str", "to_string"]
        .iter()
        .any(|k| lower.contains(k))
    {
        actions.push("serializes or parses structured data".to_string());
    }

    if ["match ", "if ", "else", "for ", "while "]
        .iter()
        .any(|k| lower.contains(k))
    {
        actions.push("contains control-flow logic".to_string());
    }

    if ["result<", "anyhow", "?;", "context("]
        .iter()
        .any(|k| lower.contains(k))
    {
        actions.push("propagates and annotates errors".to_string());
    }

    actions
}
