use crate::config::EmbeddingsConfig;
use crate::embeddings::cache::EmbeddingCache;
use crate::indexing::CodeChunk;
use anyhow::{ensure, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;

const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";
const MAX_EMBEDDING_INPUTS_PER_REQUEST: usize = 2048;

pub struct Embedder {
    client: Client,
    api_key: Option<String>,
    model: String,
    dimensions: usize,
    batch_size: usize,
    max_concurrency: usize,
}

#[derive(Debug, Serialize)]
struct OpenAiEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingItem {
    embedding: Vec<f32>,
}

impl Embedder {
    pub fn new(config: &EmbeddingsConfig) -> Result<Self> {
        ensure!(
            config.dimensions > 0,
            "embedding dimensions must be positive"
        );
        Ok(Self {
            client: Client::new(),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            model: config.model.clone(),
            dimensions: config.dimensions as usize,
            batch_size: config
                .batch_size
                .min(MAX_EMBEDDING_INPUTS_PER_REQUEST)
                .max(1),
            max_concurrency: config.max_concurrency.max(1),
        })
    }

    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if self.api_key.is_none() {
            return Ok(texts
                .iter()
                .map(|t| deterministic_embedding(t, self.dimensions))
                .collect());
        }

        let request = OpenAiEmbeddingRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };

        let response: OpenAiEmbeddingResponse = self
            .client
            .post(OPENAI_EMBEDDINGS_URL)
            .bearer_auth(self.api_key.as_deref().unwrap_or_default())
            .json(&request)
            .send()
            .await
            .context("failed to send embeddings request")?
            .error_for_status()
            .context("embeddings API returned error status")?
            .json()
            .await
            .context("failed to parse embeddings response")?;

        Ok(response
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect())
    }

    pub async fn embed_batch_chunked(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let semaphore = Arc::new(Semaphore::new(self.max_concurrency));
        let mut handles = Vec::new();

        for (idx, chunk) in texts.chunks(self.batch_size).enumerate() {
            let permit = semaphore.clone().acquire_owned().await?;
            let local_texts: Vec<String> = chunk.to_vec();
            let model = self.model.clone();
            let api_key = self.api_key.clone();
            let client = self.client.clone();
            let dimensions = self.dimensions;

            handles.push(tokio::spawn(async move {
                let _permit = permit;
                if api_key.is_none() {
                    let vectors = local_texts
                        .iter()
                        .map(|text| deterministic_embedding(text, dimensions))
                        .collect::<Vec<_>>();
                    return Ok::<(usize, Vec<Vec<f32>>), anyhow::Error>((idx, vectors));
                }

                let response: OpenAiEmbeddingResponse = client
                    .post(OPENAI_EMBEDDINGS_URL)
                    .bearer_auth(api_key.as_deref().unwrap_or_default())
                    .json(&OpenAiEmbeddingRequest {
                        model,
                        input: local_texts,
                    })
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;

                Ok((
                    idx,
                    response
                        .data
                        .into_iter()
                        .map(|item| item.embedding)
                        .collect::<Vec<_>>(),
                ))
            }));
        }

        let mut completed = Vec::with_capacity(handles.len());
        for handle in handles {
            completed.push(handle.await??);
        }
        completed.sort_by_key(|(idx, _)| *idx);

        let mut embeddings = Vec::with_capacity(texts.len());
        for (_, chunk_embeddings) in completed {
            embeddings.extend(chunk_embeddings);
        }

        Ok(embeddings)
    }

    pub async fn embed_chunks_with_cache(
        &self,
        chunks: &[CodeChunk],
        cache: &EmbeddingCache,
    ) -> Result<Vec<Vec<f32>>> {
        let mut output = vec![Vec::<f32>::new(); chunks.len()];
        let mut misses = Vec::new();
        let mut miss_positions = Vec::new();
        let mut miss_keys = Vec::new();

        for (idx, chunk) in chunks.iter().enumerate() {
            let key = self.cache_key_for_chunk(chunk);
            if let Some(embedding) = cache.get(&key).filter(|e| e.len() == self.dimensions) {
                output[idx] = embedding;
            } else {
                misses.push(chunk.code.clone());
                miss_positions.push(idx);
                miss_keys.push(key);
            }
        }

        if !misses.is_empty() {
            let new_embeddings = self.embed_batch_chunked(&misses).await?;
            for ((pos, key), embedding) in miss_positions
                .into_iter()
                .zip(miss_keys.into_iter())
                .zip(new_embeddings.into_iter())
            {
                cache.insert(key, embedding.clone());
                output[pos] = embedding;
            }
        }

        Ok(output)
    }

    fn cache_key_for_chunk(&self, chunk: &CodeChunk) -> String {
        format!(
            "model:{}:dim:{}:{}",
            self.model,
            self.dimensions,
            EmbeddingCache::key_for_chunk(chunk)
        )
    }
}

fn deterministic_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dimensions];
    if dimensions == 0 {
        return v;
    }

    for (i, byte) in text.as_bytes().iter().enumerate() {
        let idx = i % dimensions;
        v[idx] += (*byte as f32) / 255.0;
    }

    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut v {
            *value /= norm;
        }
    }

    v
}
