//! Placeholder Rust crate for `woollama`. The Python implementation in
//! `src/woollama/` is the working v0.1; this stub exists only to claim the
//! crates.io name so it doesn't get squatted before the Rust rewrite lands.
//!
//! See `docs/architecture.md` for the design.

/// The version of the woollama project (matches Python's `__version__`).
pub const VERSION: &str = "0.0.1";

/// Returns the placeholder marker for the Rust crate. The real implementation
/// will land alongside (or replace) the Python one as the architecture
/// stabilizes.
pub fn placeholder() -> &'static str {
    "woollama — Python implementation in src/woollama/. Rust comes later."
}
