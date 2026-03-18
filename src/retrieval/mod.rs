pub mod vector_store;

use crate::retrieval::vector_store::SearchHit;

pub fn rewrite_query(query: &str) -> Vec<String> {
	let normalized = query.trim();
	if normalized.is_empty() {
		return Vec::new();
	}

	let mut variants = vec![normalized.to_string()];
	variants.push(normalized.to_ascii_lowercase());

	let tokens = tokenize(normalized);
	if !tokens.is_empty() {
		variants.push(tokens.join(" "));
	}

	let keyword_focus = tokens
		.iter()
		.filter(|t| t.len() > 2)
		.take(6)
		.cloned()
		.collect::<Vec<_>>();
	if !keyword_focus.is_empty() {
		variants.push(keyword_focus.join(" "));
	}

	variants.sort();
	variants.dedup();
	variants
}

pub fn hybrid_rank_hits(query: &str, mut hits: Vec<SearchHit>, vector_weight: f32) -> Vec<SearchHit> {
	let vector_weight = vector_weight.clamp(0.0, 1.0);
	let lexical_weight = 1.0 - vector_weight;

	for hit in &mut hits {
		let lexical = lexical_score(query, &hit.chunk.code);
		hit.score = vector_weight * hit.score + lexical_weight * lexical;
	}

	hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
	hits
}

pub fn lexical_score(query: &str, text: &str) -> f32 {
	let q = tokenize(query);
	let t = tokenize(text);
	if q.is_empty() || t.is_empty() {
		return 0.0;
	}

	let t_set = t.iter().cloned().collect::<std::collections::HashSet<_>>();
	let overlap = q.iter().filter(|token| t_set.contains(*token)).count();
	overlap as f32 / q.len() as f32
}

fn tokenize(input: &str) -> Vec<String> {
	input
		.split(|c: char| !c.is_alphanumeric() && c != '_')
		.filter(|s| !s.is_empty())
		.map(|s| s.to_ascii_lowercase())
		.collect()
}
