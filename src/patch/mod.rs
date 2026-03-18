//! High-level patch utilities for repository modifications.
//!
//! This module exposes submodules that apply, validate, and revert unified
//! diffs against a repository working tree. The `apply` submodule implements
//! safe application strategies (try libgit2 then fall back to the git CLI),
//! while the `revert` submodule is responsible for undoing applied changes
//! when test/validation fails. Use these helpers from the orchestrator and
//! agents to keep patching logic centralized and well-documented.
pub mod apply;
pub mod revert;
