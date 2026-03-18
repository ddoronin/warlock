# 🧙 Warlock

Warlock is an autonomous coding agent for real repositories.

It can:

- index a codebase into semantic chunks,
- generate a multi-step implementation plan,
- create and apply patches,
- run compile/tests locally or in a Docker sandbox,
- reflect on failures and retry.

The main loop is:

```
goal → plan → retrieve context → patch code → run checks/tests → reflect/retry
```

---

## What it can do today

Warlock currently exposes 4 CLI modes:

- `index`: parse and chunk a repository.
- `plan`: generate a structured plan for a goal.
- `solve`: run the full autonomous workflow.
- `search`: query semantically indexed chunks from Qdrant.

> Note: by default, `solve` runs `cargo check` and `cargo test` in local mode (same repo/container). Docker mode is also available.

---

## Current architecture

```
src/
├── agents/
│   ├── planner.rs      # Creates structured step plans
│   ├── coder.rs        # Produces code patches
│   └── reflector.rs    # Suggests corrected patches on failures
├── embeddings/
│   ├── embedder.rs     # Embedding generation + batching
│   └── cache.rs        # Persistent embedding cache
├── indexing/
│   ├── walker.rs       # Repository traversal
│   ├── parser.rs       # Language-aware parsing
│   └── chunker.rs      # Chunk extraction
├── retrieval/
│   └── vector_store.rs # Qdrant-backed semantic search
├── patch/
│   ├── apply.rs        # Apply diffs using git2/git apply
│   └── revert.rs       # Revert failed patch attempts
├── sandbox/
│   ├── local.rs        # Local in-repo command execution
│   └── docker.rs       # Docker sandbox execution
├── orchestrator/
│   └── workflow.rs     # End-to-end solve orchestration
└── main.rs             # CLI entrypoint
```

---

## Prerequisites

- Rust toolchain (stable)
- Docker Desktop / Docker Engine (only required for `sandbox.backend = "docker"`)
- A running Qdrant instance
- LLM credentials (default config uses OpenAI)

---

## Configuration

Default config file: `warlock.toml`

Important fields:

- `[llm]` provider/model/token settings
- `[embeddings]` provider/model/cache settings
- `[vector_store]` Qdrant URL + collection
- `[sandbox]` execution backend (`local` or `docker`) and limits
- `[agent]` reflection and planning limits

Environment variables used by providers:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OLLAMA_BASE_URL`
- `QDRANT_URL` (optional override for `[vector_store].url`)
- `WARLOCK_QDRANT_URL` (same as `QDRANT_URL`, higher priority)
- `WARLOCK_QDRANT_COLLECTION` (optional override for `[vector_store].collection`)
- `WARLOCK_QDRANT_TOP_K` (optional override for `[vector_store].top_k`)

---

## Run locally

1. Create `.env` (or copy from `.env.example`) and set your keys:

```bash
OPENAI_API_KEY="<your-key>"
# Optional local overrides:
# QDRANT_URL="http://localhost:6333"
# RUST_LOG="info"
```

2. Start Qdrant:

```bash
docker compose up -d qdrant
```

3. Check CLI help:

```bash
cargo run -- --help
```

4. Run commands against a repository:

```bash
# Index repository
cargo run -- index /absolute/path/to/repo

# Generate plan
cargo run -- plan "add structured logging to service layer" --repo /absolute/path/to/repo

# Full autonomous solve
cargo run -- solve "add structured logging to service layer" --repo /absolute/path/to/repo

# Full autonomous solve with explicit Docker backend override
cargo run -- solve "add structured logging to service layer" --repo /absolute/path/to/repo --sandbox-backend docker

# Semantic search (returns CodeChunk[])
cargo run -- search "where is planner max steps used" --repo /absolute/path/to/repo

# Semantic search with similarity scores
cargo run -- search "where is planner max steps used" --repo /absolute/path/to/repo --with-scores
```

Use another config file if needed:

```bash
cargo run -- --config /path/to/warlock.toml solve "your goal" --repo /absolute/path/to/repo
```

---

## Run with Docker Compose

This project uses Docker Compose only for Qdrant when running Warlock locally.

Start Qdrant:

```bash
docker compose up -d qdrant
```

---

## Practical notes

- The target repository should be a Git repository (patch apply/check uses Git tooling).
- Default sandbox backend is `local` (runs in current codebase/container).
- Docker backend can be enabled via config (`[sandbox].backend = "docker"`) or `--sandbox-backend docker`.
- Embeddings can be cached under `.warlock/embeddings_cache` for faster repeated runs.
- Search data is persisted in Qdrant and scoped by repository identity.

---

## Status

Warlock is an active prototype focused on reliable agent loops and repository-aware change automation.
