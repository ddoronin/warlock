Canonical list of modules that need module-level summaries
=========================================================

This file records the canonical set of code modules (and a few related files)
that currently lack a module-level summary comment and should receive one.
The list was produced by inventorying the repository's module declarations and
existing top-level documentation.

For each entry below, add a module-level doc comment (//! or /*! */) at the top
of the module's primary source file describing the module's responsibility and
high-level behavior in one or two sentences.

Recommended modules to document
------------------------------
- src/coder
  - Why: Public module declared in the crate (pub mod coder;) with no recorded
    module-level summary. Describe the role of encoding/serialization helpers
    or code-generation utilities contained here.

- src/planner
  - Why: Public module declared in the crate (pub mod planner;) with no
    recorded module-level summary. Summarize the planning/summary construction
    responsibilities (e.g., building planning summaries, orchestration).

- src/cache
  - Why: Public module declared in the crate (pub mod cache;) with no
    recorded module-level summary. Describe caching strategies / persisted
    index artifacts.

- src/embedder
  - Why: Public module declared in the crate (pub mod embedder;) with no
    recorded module-level summary. Summarize embedding operations / model
    interactions.

- src/parser
  - Why: Multiple parser module declarations found (pub mod parser; repeated)
    but no canonical module-level summary recorded. Describe parsing goals,
    supported languages/formats, and produced AST/chunk shapes.

- src/indexing (src/indexing/mod.rs)
  - Why: Contains key types and functions (ModuleSummary, build_module_summaries,
    derive_module_name, build_planning_summary) but currently lacks a clear
    top-level summary. Add a concise description of the indexing subsystem,
    its inputs/outputs, and its role in building module & symbol summaries.

Related files to consider
-------------------------
- tools/verify_module_docs.rs
  - Why: Tool that verifies module docs exists and performs parsing of doc
    comments. It already contains top-level code, but documenting the tool's
    purpose makes maintenance easier.

- tests/indexing_tests.rs
  - Why: Test modules can benefit from a brief module doc explaining the
    high-level scenarios covered (grouping, serialization expectations).

Notes
-----
- Keep module summaries short and focused (1-3 sentences).
- Prefer "what" and "why" over implementation details.
- For public/published modules, ensure the summary helps new contributors and
  external readers understand the module boundary and responsibilities.

If a file on this list already has a doc comment, consider whether it is:
- Absent at the module root (e.g., inside a nested file instead of mod.rs), or
- Insufficiently descriptive and in need of an improved, canonical summary.

