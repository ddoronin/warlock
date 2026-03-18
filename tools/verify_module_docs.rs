use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

fn main() {
    // Try to read tools/module_doc_gaps.json from repository root (or CWD).
    let gaps_path = Path::new("tools/module_doc_gaps.json");
    let list = match fs::read_to_string(gaps_path) {
        Ok(s) => parse_json_string_array(&s)
            .or_else(|| parse_newline_list(&s))
            .unwrap_or_default(),
        Err(_) => {
            eprintln!(
                "warning: {} not found; nothing to verify",
                gaps_path.display()
            );
            std::process::exit(0);
        }
    };

    if list.is_empty() {
        eprintln!("no module paths found in tools/module_doc_gaps.json; nothing to verify");
        std::process::exit(0);
    }

    let mut failed = false;
    for rel in list {
        let path = Path::new(&rel);
        match fs::read_to_string(path) {
            Ok(src) => {
                let (doc_text, has_doc) = extract_top_doc_comment(&src);
                if !has_doc {
                    eprintln!("FAIL: {}: missing top-of-file module doc comment", rel);
                    failed = true;
                    continue;
                }

                let word_count = doc_text.split_whitespace().count();
                if word_count < 8 {
                    eprintln!(
                        "FAIL: {}: doc too short ({} words); should be at least 8 words",
                        rel, word_count
                    );
                    failed = true;
                }

                let exported = find_exported_type_names(&src);
                if !exported.is_empty() {
                    let doc_lower = doc_text.to_lowercase();
                    let mut found = false;
                    for name in &exported {
                        if doc_lower.contains(&name.to_lowercase()) {
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        eprintln!(
                            "FAIL: {}: doc does not mention any exported type (exported: {:?})",
                            rel, exported
                        );
                        failed = true;
                    }
                } else {
                    // No exported types found; warn but don't fail on that check.
                    eprintln!("note: {}: no exported types found in file; only word-count checked", rel);
                }
            }
            Err(e) => {
                eprintln!("FAIL: {}: could not read file: {}", rel, e);
                failed = true;
            }
        }
    }

    if failed {
        eprintln!("module docs verification failed");
        std::process::exit(2);
    } else {
        println!("module docs verification passed");
        std::process::exit(0);
    }
}

// Very small helper to parse a JSON array of strings without pulling in serde.
fn parse_json_string_array(s: &str) -> Option<Vec<String>> {
    let s = s.trim();
    if !s.starts_with('[') {
        return None;
    }
    let mut out = Vec::new();
    let mut chars = s.chars().enumerate();
    let mut in_str = false;
    let mut buf = String::new();
    let mut escaped = false;
    while let Some((_i, ch)) = chars.next() {
        if !in_str {
            if ch == '"' {
                in_str = true;
                buf.clear();
                escaped = false;
            } else {
                // skip until next quote
            }
        } else {
            if escaped {
                buf.push(match ch {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    other => other,
                });
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_str = false;
                out.push(buf.clone());
            } else {
                buf.push(ch);
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

// Fallback: parse as newline-separated list (trim lines, ignore empties and comments).
fn parse_newline_list(s: &str) -> Option<Vec<String>> {
    let mut v = Vec::new();
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        v.push(t.to_string());
    }
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

// Extract a contiguous top-of-file doc comment block and return its text (joined).
// Recognizes line doc comments starting with //! or /// and block doc comments /*! ... */ or /** ... */
fn extract_top_doc_comment(src: &str) -> (String, bool) {
    let mut lines = src.lines();
    let mut collected = Vec::new();

    // Skip initial shebang
    if let Some(l) = lines.clone().next() {
        if l.starts_with("#!") {
            // consume first line
            lines.next();
        }
    }

    // Peek first non-empty line to decide if it's a doc comment or not.
    loop {
        match lines.next() {
            Some(l) => {
                let t = l.trim_start();
                if t.is_empty() {
                    // continue scanning
                    continue;
                } else if t.starts_with("//!") || t.starts_with("///") {
                    // collect a block of contiguous doc lines (/// or //!)
                    let mut text_line = strip_line_doc_marker(t);
                    collected.push(text_line);
                    // consume following contiguous doc lines
                    for l2 in &mut lines {
                        let t2 = l2.trim_start();
                        if t2.starts_with("//!") || t2.starts_with("///") {
                            collected.push(strip_line_doc_marker(t2));
                        } else if t2.trim().is_empty() {
                            // allow blank lines in doc block
                            collected.push(String::new());
                        } else {
                            break;
                        }
                    }
                    break;
                } else if t.starts_with("/*!") || t.starts_with("/**") {
                    // block doc comment - collect until closing */
                    let mut in_block = true;
                    let mut partial = t.to_string();
                    // remove opening marker
                    partial = partial
                        .trim_start_matches("/*!")
                        .trim_start_matches("/**")
                        .to_string();
                    // if it contains closing */
                    if partial.contains("*/") {
                        let before = partial.split("*/").next().unwrap_or("").to_string();
                        collected.push(before.trim().to_string());
                        in_block = false;
                    } else {
                        collected.push(partial);
                    }
                    if in_block {
                        for l2 in &mut lines {
                            if let Some(pos) = l2.find("*/") {
                                let before = &l2[..pos];
                                collected.push(before.trim().to_string());
                                break;
                            } else {
                                collected.push(l2.trim().to_string());
                            }
                        }
                    }
                    break;
                } else {
                    // first meaningful line is not a doc comment
                    return (String::new(), false);
                }
            }
            None => return (String::new(), false),
        }
    }

    let text = collected
        .into_iter()
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    (text, !text.trim().is_empty())
}

fn strip_line_doc_marker(s: &str) -> String {
    let s = s.trim_start();
    if s.starts_with("///") {
        s.trim_start_matches("///").trim_start().to_string()
    } else if s.starts_with("//!") {
        s.trim_start_matches("//!").trim_start().to_string()
    } else {
        s.to_string()
    }
}

fn find_exported_type_names(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    // Very simple scanning; handle patterns like:
    // pub struct Name
    // pub(crate) enum Name
    // pub trait Name
    // pub type Name = ...
    for line in src.lines() {
        let t = line.trim_start();
        if !t.starts_with("pub") {
            continue;
        }
        // Remove leading "pub" and possible "(...)" qualifiers
        let mut rest = t["pub".len()..].trim_start();
        if rest.starts_with('(') {
            if let Some(pos) = rest.find(')') {
                rest = rest[pos + 1..].trim_start();
            }
        }
        // rest should now start with kind
        for kind in &["struct", "enum", "trait", "type"] {
            if rest.starts_with(kind) {
                let after = rest[kind.len()..].trim_start();
                // extract identifier
                if let Some(name) = after.split_whitespace().next() {
                    // strip generics or where clauses or '='
                    let name = name.trim_end_matches(|c: char| c == '{' || c == '<' || c == '=');
                    // remove trailing punctuation
                    let name = name.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_');
                    if !name.is_empty() && name.chars().next().unwrap().is_alphabetic() {
                        out.push(name.to_string());
                    }
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}
