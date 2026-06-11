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
    eprintln!(
        "woollama-server {} — stub (slice 0). See docs/rust-router-port.md.",
        env!("CARGO_PKG_VERSION")
    );
}
