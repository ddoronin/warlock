use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::parser::{ParsedFile, SourceLanguage};

/// A semantic unit extracted from a source file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeChunk {
    /// Relative path from repository root.
    pub file: PathBuf,
    /// Fully qualified symbol or best-effort local name.
    pub symbol: String,
    /// Symbol category for retrieval filtering.
    pub kind: SymbolKind,
    /// Extracted source code for this chunk.
    pub code: String,
    /// Byte range in source file: (start, end).
    pub span: (usize, usize),
    /// AST subtree S-expression from tree-sitter.
    pub ast_sexp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Module,
    Class,
    Method,
    Unknown,
}

pub fn extract_chunks(parsed: &ParsedFile, max_chunk_lines: usize) -> Vec<CodeChunk> {
    if is_generated_file(&parsed.source) {
        return Vec::new();
    }

    if parsed.tree.is_none() {
        return fallback_line_chunks(parsed, max_chunk_lines);
    }

    let tree = parsed.tree.as_ref().expect("checked is_some");
    let root = tree.root_node();
    let mut cursor = root.walk();
    let mut chunks = Vec::new();

    for node in root.children(&mut cursor) {
        collect_symbol_chunks(parsed, node, max_chunk_lines, &mut chunks);
    }

    if chunks.is_empty() {
        return fallback_line_chunks(parsed, max_chunk_lines);
    }

    chunks
}

fn collect_symbol_chunks(
    parsed: &ParsedFile,
    node: tree_sitter::Node<'_>,
    max_chunk_lines: usize,
    out: &mut Vec<CodeChunk>,
) {
    let kind = map_symbol_kind(parsed.language, node.kind());
    if kind != SymbolKind::Unknown {
        let start = node.start_byte();
        let end = node.end_byte();
        if start < end && end <= parsed.source.len() {
            let code = parsed.source[start..end].to_string();
            let symbol = extract_symbol_name(node, &parsed.source).unwrap_or_else(|| {
                format!(
                    "{}:{}",
                    parsed.file.display(),
                    node.start_position().row + 1
                )
            });
            let chunk = CodeChunk {
                file: parsed.file.clone(),
                symbol,
                kind,
                code,
                span: (start, end),
                ast_sexp: node.to_sexp(),
            };

            out.extend(split_large_chunk(chunk, max_chunk_lines));
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_symbol_chunks(parsed, child, max_chunk_lines, out);
    }
}

fn extract_symbol_name(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    if let Some(name_node) = node.child_by_field_name("name") {
        let start = name_node.start_byte();
        let end = name_node.end_byte();
        if start < end && end <= source.len() {
            return Some(source[start..end].trim().to_string());
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            let start = child.start_byte();
            let end = child.end_byte();
            if start < end && end <= source.len() {
                return Some(source[start..end].trim().to_string());
            }
        }
    }

    None
}

fn map_symbol_kind(language: SourceLanguage, node_kind: &str) -> SymbolKind {
    match language {
        SourceLanguage::Rust => match node_kind {
            "function_item" => SymbolKind::Function,
            "struct_item" => SymbolKind::Struct,
            "enum_item" => SymbolKind::Enum,
            "trait_item" => SymbolKind::Trait,
            "impl_item" => SymbolKind::Impl,
            "mod_item" => SymbolKind::Module,
            _ => SymbolKind::Unknown,
        },
        SourceLanguage::Python => match node_kind {
            "function_definition" => SymbolKind::Function,
            "class_definition" => SymbolKind::Class,
            _ => SymbolKind::Unknown,
        },
        SourceLanguage::TypeScript | SourceLanguage::Tsx => match node_kind {
            "function_declaration" => SymbolKind::Function,
            "method_definition" => SymbolKind::Method,
            "class_declaration" => SymbolKind::Class,
            "interface_declaration" => SymbolKind::Trait,
            "enum_declaration" => SymbolKind::Enum,
            _ => SymbolKind::Unknown,
        },
    }
}

fn split_large_chunk(chunk: CodeChunk, max_chunk_lines: usize) -> Vec<CodeChunk> {
    let total_lines = chunk.code.lines().count();
    if total_lines <= max_chunk_lines || max_chunk_lines == 0 {
        return vec![chunk];
    }

    let lines: Vec<&str> = chunk.code.lines().collect();
    let mut line_offsets = Vec::with_capacity(lines.len() + 1);
    line_offsets.push(0usize);
    for line in &lines {
        let next = line_offsets.last().copied().unwrap_or(0) + line.len() + 1;
        line_offsets.push(next);
    }

    let mut parts = Vec::new();
    let mut start_line = 0usize;
    let mut part_idx = 1usize;
    while start_line < lines.len() {
        let end_line = choose_split_boundary(&lines, start_line, max_chunk_lines);
        let part_code = lines[start_line..end_line].join("\n");

        let start_offset = line_offsets[start_line];
        let end_offset = line_offsets[end_line].saturating_sub(1);
        let start_byte = chunk.span.0 + start_offset;
        let end_byte = (chunk.span.0 + end_offset).min(chunk.span.1);

        parts.push(CodeChunk {
            file: chunk.file.clone(),
            symbol: format!("{}#part{}", chunk.symbol, part_idx),
            kind: chunk.kind.clone(),
            code: part_code,
            span: (start_byte, end_byte),
            ast_sexp: chunk.ast_sexp.clone(),
        });

        part_idx += 1;
        start_line = end_line;
    }

    parts
}

fn choose_split_boundary(lines: &[&str], start: usize, max_chunk_lines: usize) -> usize {
    let hard_end = (start + max_chunk_lines).min(lines.len());
    if hard_end == lines.len() {
        return hard_end;
    }

    let soft_start = start + (max_chunk_lines * 3 / 5);
    let soft_start = soft_start.min(hard_end.saturating_sub(1));
    for idx in (soft_start..hard_end).rev() {
        let trimmed = lines[idx].trim();
        if trimmed == "}" || trimmed == "];" || trimmed == ")" || trimmed.ends_with(':') {
            return idx + 1;
        }
    }

    hard_end
}

fn fallback_line_chunks(parsed: &ParsedFile, max_chunk_lines: usize) -> Vec<CodeChunk> {
    if max_chunk_lines == 0 {
        return Vec::new();
    }

    let lines: Vec<&str> = parsed.source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut line_offsets = Vec::with_capacity(lines.len() + 1);
    line_offsets.push(0usize);
    for line in &lines {
        let next = line_offsets.last().copied().unwrap_or(0) + line.len() + 1;
        line_offsets.push(next);
    }

    let mut chunk_idx = 0usize;
    let mut start = 0usize;
    while start < lines.len() {
        let end = (start + max_chunk_lines).min(lines.len());
        let code = lines[start..end].join("\n");

        let start_byte = line_offsets[start];
        let mut end_byte = line_offsets[end].saturating_sub(1);
        if end == lines.len() {
            end_byte = parsed.source.len();
        }

        chunk_idx += 1;
        chunks.push(CodeChunk {
            file: parsed.file.clone(),
            symbol: format!("{}#fallback{}", parsed.file.display(), chunk_idx),
            kind: SymbolKind::Unknown,
            code,
            span: (start_byte, end_byte),
            ast_sexp: "(fallback_line_chunk)".to_string(),
        });

        start = end;
    }

    chunks
}

fn is_generated_file(source: &str) -> bool {
    source.lines().take(50).any(|line| {
        let lower = line.to_ascii_lowercase();
        lower.contains("@generated")
            || lower.contains("auto-generated")
            || lower.contains("autogenerated")
            || lower.contains("generated file") && lower.contains("do not edit")
    })
}
