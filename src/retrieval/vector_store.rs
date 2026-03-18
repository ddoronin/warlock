use crate::indexing::{CodeChunk, FileSummary, SymbolKind, SymbolSummary};
use anyhow::{bail, ensure, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, warn};
use uuid::Uuid;

const VECTOR_SCHEMA_VERSION: u64 = 1;

#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    pub file: Option<String>,
    pub language: Option<String>,
    pub symbol_kind: Option<SymbolKind>,
    pub doc_type: Option<VectorDocType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub score: f32,
    pub chunk: CodeChunk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarySearchHit {
    pub score: f32,
    pub doc_type: VectorDocType,
    pub file: String,
    pub symbol: Option<String>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VectorDocType {
    CodeChunk,
    SymbolSummary,
    FileSummary,
    Documentation,
}

impl Default for VectorDocType {
    fn default() -> Self {
        Self::CodeChunk
    }
}

pub struct VectorStore {
    client: Client,
    url: String,
    collection: String,
    dim: usize,
}

impl VectorStore {
    pub async fn new(url: &str, collection: &str, dim: u64) -> Result<Self> {
        ensure!(dim > 0, "vector dimension must be positive");

        let store = Self {
            client: Client::new(),
            url: normalize_base_url(url),
            collection: collection.to_string(),
            dim: dim as usize,
        };

        store.ensure_collection().await?;
        store.ensure_payload_indexes().await?;
        Ok(store)
    }

    pub async fn upsert(
        &self,
        repo_id: &str,
        chunks: &[CodeChunk],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        ensure!(!repo_id.trim().is_empty(), "repo_id must not be empty");
        ensure!(
            chunks.len() == embeddings.len(),
            "chunks/embeddings length mismatch"
        );

        info!(
            repo_id = repo_id,
            chunks = chunks.len(),
            "qdrant upsert code chunks start"
        );

        for batch in chunks
            .iter()
            .zip(embeddings.iter())
            .collect::<Vec<_>>()
            .chunks(128)
        {
            let mut points = Vec::with_capacity(batch.len());
            for (chunk, embedding) in batch {
                ensure!(
                    embedding.len() == self.dim,
                    "embedding dimension mismatch: expected {}, got {}",
                    self.dim,
                    embedding.len()
                );

                points.push(json!({
                    "id": point_id(repo_id, chunk),
                    "vector": embedding,
                    "payload": {
                        "schema_version": VECTOR_SCHEMA_VERSION,
                        "doc_type": doc_type_to_string(&VectorDocType::CodeChunk),
                        "repo_id": repo_id,
                        "file": chunk.file.to_string_lossy().to_string(),
                        "symbol": chunk.symbol,
                        "symbol_kind": symbol_kind_to_string(&chunk.kind),
                        "kind": symbol_kind_to_string(&chunk.kind),
                        "code": chunk.code,
                        "span_start": chunk.span.0,
                        "span_end": chunk.span.1,
                        "ast_sexp": chunk.ast_sexp,
                        "language": language_from_file(chunk),
                    }
                }));
            }

            let body = json!({ "points": points });
            debug!(
                repo_id = repo_id,
                points = batch.len(),
                "qdrant upsert code chunks batch"
            );
            self.put_json(
                &format!("/collections/{}/points?wait=true", self.collection),
                &body,
            )
            .await
            .context("qdrant upsert failed")?;
        }

        info!(
            repo_id = repo_id,
            chunks = chunks.len(),
            "qdrant upsert code chunks complete"
        );

        Ok(())
    }

    pub async fn upsert_file_summaries(
        &self,
        repo_id: &str,
        summaries: &[FileSummary],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        ensure!(!repo_id.trim().is_empty(), "repo_id must not be empty");
        ensure!(
            summaries.len() == embeddings.len(),
            "summaries/embeddings length mismatch"
        );

        info!(
            repo_id = repo_id,
            summaries = summaries.len(),
            "qdrant upsert file summaries start"
        );

        for batch in summaries
            .iter()
            .zip(embeddings.iter())
            .collect::<Vec<_>>()
            .chunks(128)
        {
            let mut points = Vec::with_capacity(batch.len());
            for (summary, embedding) in batch {
                ensure!(
                    embedding.len() == self.dim,
                    "embedding dimension mismatch: expected {}, got {}",
                    self.dim,
                    embedding.len()
                );

                points.push(json!({
                    "id": point_id_for_file_summary(repo_id, summary),
                    "vector": embedding,
                    "payload": {
                        "schema_version": VECTOR_SCHEMA_VERSION,
                        "doc_type": doc_type_to_string(&VectorDocType::FileSummary),
                        "repo_id": repo_id,
                        "file": summary.file.to_string_lossy().to_string(),
                        "language": language_from_path(&summary.file),
                        "summary": summary.summary,
                    }
                }));
            }

            let body = json!({ "points": points });
            self.put_json(
                &format!("/collections/{}/points?wait=true", self.collection),
                &body,
            )
            .await
            .context("qdrant upsert file summaries failed")?;
        }

        info!(
            repo_id = repo_id,
            summaries = summaries.len(),
            "qdrant upsert file summaries complete"
        );

        Ok(())
    }

    pub async fn upsert_symbol_summaries(
        &self,
        repo_id: &str,
        summaries: &[SymbolSummary],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        ensure!(!repo_id.trim().is_empty(), "repo_id must not be empty");
        ensure!(
            summaries.len() == embeddings.len(),
            "summaries/embeddings length mismatch"
        );

        info!(
            repo_id = repo_id,
            summaries = summaries.len(),
            "qdrant upsert symbol summaries start"
        );

        for batch in summaries
            .iter()
            .zip(embeddings.iter())
            .collect::<Vec<_>>()
            .chunks(128)
        {
            let mut points = Vec::with_capacity(batch.len());
            for (summary, embedding) in batch {
                ensure!(
                    embedding.len() == self.dim,
                    "embedding dimension mismatch: expected {}, got {}",
                    self.dim,
                    embedding.len()
                );

                points.push(json!({
                    "id": point_id_for_symbol_summary(repo_id, summary),
                    "vector": embedding,
                    "payload": {
                        "schema_version": VECTOR_SCHEMA_VERSION,
                        "doc_type": doc_type_to_string(&VectorDocType::SymbolSummary),
                        "repo_id": repo_id,
                        "file": summary.file.to_string_lossy().to_string(),
                        "symbol": summary.symbol,
                        "symbol_kind": symbol_kind_to_string(&summary.kind),
                        "kind": symbol_kind_to_string(&summary.kind),
                        "language": language_from_path(&summary.file),
                        "summary": summary.summary,
                    }
                }));
            }

            let body = json!({ "points": points });
            self.put_json(
                &format!("/collections/{}/points?wait=true", self.collection),
                &body,
            )
            .await
            .context("qdrant upsert symbol summaries failed")?;
        }

        info!(
            repo_id = repo_id,
            summaries = summaries.len(),
            "qdrant upsert symbol summaries complete"
        );

        Ok(())
    }

    pub async fn search(
        &self,
        repo_id: &str,
        query_vec: Vec<f32>,
        top_k: usize,
        filter: Option<&SearchFilter>,
    ) -> Result<Vec<CodeChunk>> {
        let hits = self
            .search_with_scores(repo_id, query_vec, top_k, filter)
            .await?;
        Ok(hits.into_iter().map(|h| h.chunk).collect())
    }

    pub async fn search_with_scores(
        &self,
        repo_id: &str,
        query_vec: Vec<f32>,
        top_k: usize,
        filter: Option<&SearchFilter>,
    ) -> Result<Vec<SearchHit>> {
        ensure!(!repo_id.trim().is_empty(), "repo_id must not be empty");
        ensure!(query_vec.len() == self.dim, "query dimension mismatch");
        ensure!(top_k > 0, "top_k must be > 0");
        debug!(repo_id = repo_id, top_k, filter = ?filter, "qdrant search start");

        let body = json!({
            "vector": query_vec,
            "limit": top_k,
            "with_payload": true,
            "filter": build_filter(repo_id, filter),
        });

        let response = self
            .post_json(
                &format!("/collections/{}/points/search", self.collection),
                &body,
            )
            .await
            .context("qdrant search failed")?;

        let results = response
            .get("result")
            .and_then(Value::as_array)
            .context("missing search result array")?;

        let mut hits = Vec::with_capacity(results.len());
        for item in results {
            let score = item
                .get("score")
                .and_then(Value::as_f64)
                .unwrap_or_default() as f32;
            let payload = item.get("payload").context("search item missing payload")?;
            hits.push(SearchHit {
                score,
                chunk: payload_to_chunk(payload)?,
            });
        }

        info!(
            repo_id = repo_id,
            top_k,
            hits = hits.len(),
            "qdrant search complete"
        );

        Ok(hits)
    }

    pub async fn search_summaries_with_scores(
        &self,
        repo_id: &str,
        query_vec: Vec<f32>,
        top_k: usize,
        doc_type: VectorDocType,
    ) -> Result<Vec<SummarySearchHit>> {
        ensure!(!repo_id.trim().is_empty(), "repo_id must not be empty");
        ensure!(query_vec.len() == self.dim, "query dimension mismatch");
        ensure!(top_k > 0, "top_k must be > 0");

        let requested_doc_type = doc_type.clone();
        debug!(repo_id = repo_id, top_k, doc_type = ?requested_doc_type, "qdrant summary search start");

        let filter = SearchFilter {
            file: None,
            language: None,
            symbol_kind: None,
            doc_type: Some(doc_type),
        };

        let body = json!({
            "vector": query_vec,
            "limit": top_k,
            "with_payload": true,
            "filter": build_filter(repo_id, Some(&filter)),
        });

        let response = self
            .post_json(
                &format!("/collections/{}/points/search", self.collection),
                &body,
            )
            .await
            .context("qdrant summary search failed")?;

        let results = response
            .get("result")
            .and_then(Value::as_array)
            .context("missing search result array")?;

        let mut hits = Vec::with_capacity(results.len());
        for item in results {
            let score = item
                .get("score")
                .and_then(Value::as_f64)
                .unwrap_or_default() as f32;
            let payload = item.get("payload").context("search item missing payload")?;

            let doc_type_raw = payload
                .get("doc_type")
                .and_then(Value::as_str)
                .unwrap_or("code_chunk");
            let parsed_doc_type: VectorDocType =
                serde_json::from_value(Value::String(doc_type_raw.to_string()))
                    .with_context(|| format!("invalid doc_type: {doc_type_raw}"))?;

            let file = payload
                .get("file")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let symbol = payload
                .get("symbol")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let summary = payload
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();

            hits.push(SummarySearchHit {
                score,
                doc_type: parsed_doc_type,
                file,
                symbol,
                summary,
            });
        }

        info!(repo_id = repo_id, top_k, doc_type = ?requested_doc_type, hits = hits.len(), "qdrant summary search complete");

        Ok(hits)
    }

    pub async fn has_code_chunks(&self, repo_id: &str) -> Result<bool> {
        ensure!(!repo_id.trim().is_empty(), "repo_id must not be empty");

        let body = json!({
            "exact": false,
            "filter": build_filter(repo_id, None),
        });

        let response = self
            .post_json(
                &format!("/collections/{}/points/count", self.collection),
                &body,
            )
            .await
            .context("qdrant count failed")?;

        let count = response
            .pointer("/result/count")
            .and_then(Value::as_u64)
            .context("missing qdrant count result")?;

        Ok(count > 0)
    }

    pub async fn has_doc_type(&self, repo_id: &str, doc_type: VectorDocType) -> Result<bool> {
        ensure!(!repo_id.trim().is_empty(), "repo_id must not be empty");

        let body = json!({
            "exact": false,
            "filter": {
                "must": [
                    {
                        "key": "repo_id",
                        "match": { "value": repo_id }
                    },
                    {
                        "key": "doc_type",
                        "match": { "value": doc_type_to_string(&doc_type) }
                    }
                ]
            },
        });

        let response = self
            .post_json(
                &format!("/collections/{}/points/count", self.collection),
                &body,
            )
            .await
            .with_context(|| format!("qdrant count for doc_type {:?} failed", doc_type))?;

        let count = response
            .pointer("/result/count")
            .and_then(Value::as_u64)
            .context("missing qdrant count result")?;

        Ok(count > 0)
    }

    pub async fn delete_code_chunks(&self, repo_id: &str) -> Result<()> {
        ensure!(!repo_id.trim().is_empty(), "repo_id must not be empty");

        let body = json!({
            "filter": build_filter(repo_id, None),
        });

        self.post_json(
            &format!("/collections/{}/points/delete?wait=true", self.collection),
            &body,
        )
        .await
        .context("qdrant delete failed")?;

        Ok(())
    }

    pub async fn delete_by_file(&self, repo_id: &str, file: &str) -> Result<()> {
        ensure!(!repo_id.trim().is_empty(), "repo_id must not be empty");
        ensure!(!file.trim().is_empty(), "file must not be empty");

        let body = json!({
            "filter": {
                "must": [
                    {
                        "key": "repo_id",
                        "match": { "value": repo_id }
                    },
                    {
                        "key": "file",
                        "match": { "value": file }
                    }
                ]
            }
        });

        self.post_json(
            &format!("/collections/{}/points/delete?wait=true", self.collection),
            &body,
        )
        .await
        .context("qdrant delete by file failed")?;

        Ok(())
    }

    pub async fn delete_doc_type(&self, repo_id: &str, doc_type: VectorDocType) -> Result<()> {
        ensure!(!repo_id.trim().is_empty(), "repo_id must not be empty");

        let body = json!({
            "filter": {
                "must": [
                    {
                        "key": "repo_id",
                        "match": { "value": repo_id }
                    },
                    {
                        "key": "doc_type",
                        "match": { "value": doc_type_to_string(&doc_type) }
                    }
                ]
            }
        });

        self.post_json(
            &format!("/collections/{}/points/delete?wait=true", self.collection),
            &body,
        )
        .await
        .with_context(|| format!("qdrant delete for doc_type {:?} failed", doc_type))?;

        Ok(())
    }

    async fn ensure_collection(&self) -> Result<()> {
        let resp = self
            .client
            .get(format!("{}/collections/{}", self.url, self.collection))
            .send()
            .await
            .context("failed to query qdrant collection")?;

        if resp.status().as_u16() == 404 {
            let body = json!({
                "vectors": {
                    "size": self.dim,
                    "distance": "Cosine"
                }
            });
            self.put_json(&format!("/collections/{}", self.collection), &body)
                .await
                .context("failed to create qdrant collection")?;
            return Ok(());
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!(
                "failed to inspect qdrant collection {} ({}): {}",
                self.collection,
                status,
                body
            );
        }

        let body: Value = resp
            .json()
            .await
            .context("failed to parse qdrant collection metadata")?;

        let vectors = body
            .pointer("/result/config/params/vectors")
            .context("collection metadata missing vectors config")?;

        let (size, distance) = extract_vector_config(vectors)?;
        if size != self.dim {
            bail!(
                "qdrant collection '{}' dimension mismatch: expected {}, found {}",
                self.collection,
                self.dim,
                size
            );
        }
        if distance.to_ascii_lowercase() != "cosine" {
            bail!(
                "qdrant collection '{}' distance mismatch: expected cosine, found {}",
                self.collection,
                distance
            );
        }

        Ok(())
    }

    async fn ensure_payload_indexes(&self) -> Result<()> {
        for field in ["repo_id", "file", "language", "symbol_kind", "doc_type"] {
            let body = json!({
                "field_name": field,
                "field_schema": "keyword"
            });
            let response = self
                .client
                .put(format!(
                    "{}/collections/{}/index",
                    self.url, self.collection
                ))
                .json(&body)
                .send()
                .await
                .with_context(|| format!("failed creating payload index for field '{field}'"))?;

            if response.status().is_success() {
                continue;
            }

            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            if text.to_ascii_lowercase().contains("already exists") {
                continue;
            }

            bail!(
                "failed creating payload index for field '{field}' ({}): {}",
                status,
                text
            );
        }

        Ok(())
    }

    async fn put_json(&self, path: &str, body: &Value) -> Result<Value> {
        let endpoint = format!("{}{}", self.url, path);
        let started = Instant::now();
        info!(method = "PUT", endpoint = %endpoint, "qdrant call");
        debug!(method = "PUT", endpoint = %endpoint, request_preview = %preview_json(body, 3000), "qdrant request body");
        let response = self
            .client
            .put(endpoint.clone())
            .json(body)
            .send()
            .await
            .with_context(|| format!("request to {}{} failed", self.url, path))?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            warn!(method = "PUT", endpoint = %endpoint, status = %status, elapsed_ms = started.elapsed().as_millis() as u64, response_preview = %truncate_preview(&text, 1500), "qdrant request failed");
            bail!(
                "request to {}{} failed ({}): {}",
                self.url,
                path,
                status,
                text
            );
        }

        let status = response.status();
        let text = response
            .text()
            .await
            .context("failed to read qdrant response body")?;

        info!(method = "PUT", endpoint = %endpoint, status = %status, elapsed_ms = started.elapsed().as_millis() as u64, "qdrant call complete");
        debug!(method = "PUT", endpoint = %endpoint, response_preview = %truncate_preview(&text, 2000), "qdrant response body");

        serde_json::from_str::<Value>(&text).context("failed to parse qdrant response")
    }

    async fn post_json(&self, path: &str, body: &Value) -> Result<Value> {
        let endpoint = format!("{}{}", self.url, path);
        let started = Instant::now();
        info!(
            method = "POST",
            endpoint = %endpoint,
            request_preview = %preview_json(body, 3000),
            "qdrant call"
        );
        debug!(method = "POST", endpoint = %endpoint, request_preview = %preview_json(body, 3000), "qdrant request body");
        let response = self
            .client
            .post(endpoint.clone())
            .json(body)
            .send()
            .await
            .with_context(|| format!("request to {}{} failed", self.url, path))?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            warn!(method = "POST", endpoint = %endpoint, status = %status, elapsed_ms = started.elapsed().as_millis() as u64, response_preview = %truncate_preview(&text, 1500), "qdrant request failed");
            bail!(
                "request to {}{} failed ({}): {}",
                self.url,
                path,
                status,
                text
            );
        }

        let status = response.status();
        let text = response
            .text()
            .await
            .context("failed to read qdrant response body")?;

        info!(method = "POST", endpoint = %endpoint, status = %status, elapsed_ms = started.elapsed().as_millis() as u64, "qdrant call complete");
        debug!(method = "POST", endpoint = %endpoint, response_preview = %truncate_preview(&text, 2000), "qdrant response body");

        serde_json::from_str::<Value>(&text).context("failed to parse qdrant response")
    }
}

fn preview_json(value: &Value, max_len: usize) -> String {
    let json = value.to_string();
    truncate_preview(&json, max_len)
}

fn truncate_preview(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    format!("{}...", &s[..max_len])
}

pub fn derive_repo_id(repo_path: &Path) -> String {
    let basis = repo_identity_basis(repo_path);

    let mut hasher = Sha256::new();
    hasher.update(basis.as_bytes());
    format!("repo_{}", hex::encode(hasher.finalize()))
}

fn repo_identity_basis(repo_path: &Path) -> String {
    if let Some(url) = repository_remote_url(repo_path) {
        return format!("remote:{}", normalize_repo_url(&url));
    }

    let canonical = repo_path
        .canonicalize()
        .unwrap_or_else(|_| repo_path.to_path_buf());
    format!("local:{}", canonical.to_string_lossy())
}

pub fn repository_remote_url(repo_path: &Path) -> Option<String> {
    let repo = git2::Repository::open(repo_path).ok()?;

    if let Ok(remote) = repo.find_remote("origin") {
        if let Some(url) = remote.url() {
            return Some(url.to_string());
        }
    }

    let remotes = repo.remotes().ok()?;
    for name in remotes.iter().flatten() {
        if let Ok(remote) = repo.find_remote(name) {
            if let Some(url) = remote.url() {
                return Some(url.to_string());
            }
        }
    }

    None
}

pub fn normalize_repo_url(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/').trim_end_matches(".git");

    if let Some(rest) = trimmed.strip_prefix("git@") {
        if let Some((host, path)) = rest.split_once(':') {
            return format!(
                "{}/{}",
                host.to_ascii_lowercase(),
                path.trim_start_matches('/').to_ascii_lowercase()
            );
        }
    }

    if let Some(rest) = trimmed.strip_prefix("ssh://") {
        let rest = rest.rsplit_once('@').map(|(_, r)| r).unwrap_or(rest);
        if let Some((host, path)) = rest.split_once('/') {
            return format!(
                "{}/{}",
                host.to_ascii_lowercase(),
                path.trim_start_matches('/').to_ascii_lowercase()
            );
        }
    }

    for prefix in ["https://", "http://"] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            if let Some((host, path)) = rest.split_once('/') {
                return format!(
                    "{}/{}",
                    host.to_ascii_lowercase(),
                    path.trim_start_matches('/').to_ascii_lowercase()
                );
            }
            return rest.to_ascii_lowercase();
        }
    }

    trimmed.to_ascii_lowercase()
}

fn normalize_base_url(url: &str) -> String {
    let mut normalized = url.trim().trim_end_matches('/').to_string();
    if normalized.ends_with(":6334") {
        normalized = normalized.trim_end_matches(":6334").to_string() + ":6333";
    }
    normalized
}

fn extract_vector_config(vectors: &Value) -> Result<(usize, String)> {
    if let Some(size) = vectors.get("size").and_then(Value::as_u64) {
        let distance = vectors
            .get("distance")
            .and_then(Value::as_str)
            .unwrap_or("cosine")
            .to_string();
        return Ok((size as usize, distance));
    }

    if let Some(named) = vectors.as_object() {
        let (_, cfg) = named.iter().next().context("empty named vector config")?;
        let size = cfg
            .get("size")
            .and_then(Value::as_u64)
            .context("named vectors config missing size")? as usize;
        let distance = cfg
            .get("distance")
            .and_then(Value::as_str)
            .unwrap_or("cosine")
            .to_string();
        return Ok((size, distance));
    }

    bail!("unsupported vectors config shape")
}

fn build_filter(repo_id: &str, filter: Option<&SearchFilter>) -> Value {
    let doc_type = filter
        .and_then(|f| f.doc_type.as_ref().cloned())
        .unwrap_or_default();

    let mut must = vec![
        json!({
            "key": "repo_id",
            "match": { "value": repo_id }
        }),
        json!({
            "key": "doc_type",
            "match": { "value": doc_type_to_string(&doc_type) }
        }),
    ];

    if let Some(filter) = filter {
        if let Some(file) = &filter.file {
            must.push(json!({
                "key": "file",
                "match": { "value": file }
            }));
        }

        if let Some(language) = &filter.language {
            must.push(json!({
                "key": "language",
                "match": { "value": language.to_ascii_lowercase() }
            }));
        }

        if let Some(kind) = &filter.symbol_kind {
            must.push(json!({
                "key": "symbol_kind",
                "match": { "value": symbol_kind_to_string(kind) }
            }));
        }
    }

    json!({ "must": must })
}

fn payload_to_chunk(payload: &Value) -> Result<CodeChunk> {
    if let Some(doc_type_raw) = payload.get("doc_type").and_then(Value::as_str) {
        let doc_type: VectorDocType =
            serde_json::from_value(Value::String(doc_type_raw.to_string()))
                .with_context(|| format!("invalid doc_type: {doc_type_raw}"))?;
        ensure!(
            doc_type == VectorDocType::CodeChunk,
            "payload doc_type must be code_chunk for chunk decode"
        );
    }

    let file = payload
        .get("file")
        .and_then(Value::as_str)
        .context("payload missing file")?;
    let symbol = payload
        .get("symbol")
        .and_then(Value::as_str)
        .context("payload missing symbol")?;
    let kind_raw = payload
        .get("symbol_kind")
        .or_else(|| payload.get("kind"))
        .and_then(Value::as_str)
        .context("payload missing symbol_kind/kind")?;
    let kind: SymbolKind = serde_json::from_value(Value::String(kind_raw.to_string()))
        .with_context(|| format!("invalid symbol kind: {kind_raw}"))?;
    let code = payload
        .get("code")
        .and_then(Value::as_str)
        .context("payload missing code")?;
    let span_start = payload
        .get("span_start")
        .and_then(Value::as_u64)
        .context("payload missing span_start")? as usize;
    let span_end = payload
        .get("span_end")
        .and_then(Value::as_u64)
        .context("payload missing span_end")? as usize;
    let ast_sexp = payload
        .get("ast_sexp")
        .and_then(Value::as_str)
        .context("payload missing ast_sexp")?;

    Ok(CodeChunk {
        file: file.into(),
        symbol: symbol.to_string(),
        kind,
        code: code.to_string(),
        span: (span_start, span_end),
        ast_sexp: ast_sexp.to_string(),
    })
}

fn point_id(repo_id: &str, chunk: &CodeChunk) -> String {
    let mut hasher = Sha256::new();
    hasher.update(repo_id.as_bytes());
    hasher.update(b"|");
    hasher.update(chunk.file.to_string_lossy().as_bytes());
    hasher.update(b"|");
    hasher.update(chunk.symbol.as_bytes());
    hasher.update(b"|");
    hasher.update(symbol_kind_to_string(&chunk.kind).as_bytes());
    hasher.update(b"|");
    hasher.update(chunk.span.0.to_string().as_bytes());
    hasher.update(b"|");
    hasher.update(chunk.span.1.to_string().as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    Uuid::from_bytes(bytes).to_string()
}

fn point_id_for_file_summary(repo_id: &str, summary: &FileSummary) -> String {
    deterministic_uuid(
        repo_id,
        &[
            "file_summary",
            &summary.file.to_string_lossy(),
            &summary.summary,
        ],
    )
}

fn point_id_for_symbol_summary(repo_id: &str, summary: &SymbolSummary) -> String {
    deterministic_uuid(
        repo_id,
        &[
            "symbol_summary",
            &summary.file.to_string_lossy(),
            &summary.symbol,
            &symbol_kind_to_string(&summary.kind),
            &summary.summary,
        ],
    )
}

fn deterministic_uuid(repo_id: &str, parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(repo_id.as_bytes());
    for part in parts {
        hasher.update(b"|");
        hasher.update(part.as_bytes());
    }
    let digest = hasher.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    Uuid::from_bytes(bytes).to_string()
}

fn language_from_file(chunk: &CodeChunk) -> String {
    language_from_path(&chunk.file)
}

fn language_from_path(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_ascii_lowercase()
}

fn symbol_kind_to_string(kind: &SymbolKind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(ToString::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

fn doc_type_to_string(doc_type: &VectorDocType) -> String {
    serde_json::to_value(doc_type)
        .ok()
        .and_then(|v| v.as_str().map(ToString::to_string))
        .unwrap_or_else(|| "code_chunk".to_string())
}

#[cfg(test)]
mod tests {
    use super::{build_filter, normalize_repo_url, payload_to_chunk, VectorDocType};
    use serde_json::json;

    #[test]
    fn build_filter_defaults_to_code_chunk_doc_type() {
        let filter = build_filter("repo_x", None);
        let must = filter
            .get("must")
            .and_then(|v| v.as_array())
            .expect("must array should exist");

        assert!(must.iter().any(|c| {
            c.get("key").and_then(|v| v.as_str()) == Some("doc_type")
                && c.pointer("/match/value").and_then(|v| v.as_str()) == Some("code_chunk")
        }));
    }

    #[test]
    fn payload_to_chunk_accepts_legacy_payload_without_doc_type() {
        let payload = json!({
            "file": "src/main.rs",
            "symbol": "main",
            "symbol_kind": "function",
            "code": "fn main() {}",
            "span_start": 1,
            "span_end": 1,
            "ast_sexp": "(function_item)"
        });

        let chunk = payload_to_chunk(&payload).expect("legacy payload should decode");
        assert_eq!(chunk.symbol, "main");
    }

    #[test]
    fn payload_to_chunk_rejects_non_code_chunk_doc_type() {
        let payload = json!({
            "doc_type": "file_summary",
            "file": "src/main.rs",
            "symbol": "main",
            "symbol_kind": "function",
            "code": "fn main() {}",
            "span_start": 1,
            "span_end": 1,
            "ast_sexp": "(function_item)"
        });

        let err = payload_to_chunk(&payload).expect_err("non code chunk should fail");
        assert!(err.to_string().contains("doc_type"));
        let _ = VectorDocType::CodeChunk;
    }

    #[test]
    fn normalizes_git_and_https_urls_to_same_basis() {
        let ssh = normalize_repo_url("git@github.com:Owner/Repo.git");
        let https = normalize_repo_url("https://github.com/owner/repo");
        assert_eq!(ssh, https);
    }
}
