use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tree_sitter::{Parser, Tree};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceLanguage {
    Rust,
    Python,
    TypeScript,
    Tsx,
}

impl SourceLanguage {
    pub fn as_config_name(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParsedFile {
    pub file: PathBuf,
    pub language: SourceLanguage,
    pub source: String,
    pub tree: Option<Tree>,
    pub parse_failed: bool,
}

pub struct LanguageParsers {
    rust: Parser,
    python: Parser,
    typescript: Parser,
    tsx: Parser,
}

impl LanguageParsers {
    pub fn new() -> Result<Self> {
        let mut rust = Parser::new();
        rust.set_language(&tree_sitter_rust::LANGUAGE.into())
            .context("failed to initialize tree-sitter rust parser")?;

        let mut python = Parser::new();
        python
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .context("failed to initialize tree-sitter python parser")?;

        let mut typescript = Parser::new();
        typescript
            .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
            .context("failed to initialize tree-sitter typescript parser")?;

        let mut tsx = Parser::new();
        tsx.set_language(&tree_sitter_typescript::LANGUAGE_TSX.into())
            .context("failed to initialize tree-sitter tsx parser")?;

        Ok(Self {
            rust,
            python,
            typescript,
            tsx,
        })
    }

    fn parser_mut(&mut self, lang: SourceLanguage) -> &mut Parser {
        match lang {
            SourceLanguage::Rust => &mut self.rust,
            SourceLanguage::Python => &mut self.python,
            SourceLanguage::TypeScript => &mut self.typescript,
            SourceLanguage::Tsx => &mut self.tsx,
        }
    }
}

pub fn detect_language(path: &Path) -> Option<SourceLanguage> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Some(SourceLanguage::Rust),
        "py" => Some(SourceLanguage::Python),
        "ts" => Some(SourceLanguage::TypeScript),
        "tsx" => Some(SourceLanguage::Tsx),
        _ => None,
    }
}

pub fn parse_file(
    repo_root: &Path,
    abs_path: &Path,
    parsers: &mut LanguageParsers,
    supported_languages: &[String],
) -> Result<Option<ParsedFile>> {
    let language = match detect_language(abs_path) {
        Some(lang) => lang,
        None => return Ok(None),
    };

    let allowed: HashSet<String> = supported_languages
        .iter()
        .map(|s| s.trim().to_ascii_lowercase())
        .collect();
    if !allowed.contains(language.as_config_name()) {
        return Ok(None);
    }

    let source = std::fs::read_to_string(abs_path)
        .with_context(|| format!("failed to read source file {}", abs_path.display()))?;
    let rel_file = abs_path.strip_prefix(repo_root).unwrap_or(abs_path).to_path_buf();

    let parser = parsers.parser_mut(language);
    let tree = parser.parse(&source, None);
    let parse_failed = tree
        .as_ref()
        .map(|t| t.root_node().has_error())
        .unwrap_or(true);

    Ok(Some(ParsedFile {
        file: rel_file,
        language,
        source,
        tree,
        parse_failed,
    }))
}
