//! The woollama router service, as a native Rust binary.
//!
//! STUB (slice 0): this crate exists so the cargo workspace hosts the eventual
//! server binary alongside the engine cdylib and the PyO3 wheel, and so we prove
//! the three build targets coexist without breaking the maturin wheel build.
//!
//! It does nothing yet. The HTTP surface (axum), the MCP aggregator (rmcp), the
//! conversation stores, and the claude-code executor land from slice 2 onward,
//! built on `woollama-core` once that is a pure rlib (slice 1).
//!
//! See docs/rust-router-port.md for the slice plan.

fn main() {
    // Slice 1 proof: the engine rlib links into a pure-Rust binary (no PyO3).
    // The HTTP surface that drives these primitives lands in slice 2.
    eprintln!(
        "woollama-server {} — stub (slice 1). engine providers: {}",
        env!("CARGO_PKG_VERSION"),
        woollama_engine::provider_names().join(", ")
    );
}
