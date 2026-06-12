//! Drift guard: the default config files vendored INTO this crate (`woollama-server/defaults/`,
//! so the crate is self-contained for a crates.io publish) must stay byte-identical to the
//! canonical Python-package copies (`src/woollama/defaults/`). config.rs `include_str!`s the
//! vendored copies; this test runs in the monorepo (where both exist) and fails on any drift.
//! In a published-crate checkout the Python tree is absent, so the check is skipped, not failed.

use std::path::Path;

fn check(name: &str, vendored: &str) {
    let py = Path::new(env!("CARGO_MANIFEST_DIR")).join("../src/woollama/defaults").join(name);
    match std::fs::read_to_string(&py) {
        Ok(python) => assert_eq!(
            vendored, python,
            "{name}: woollama-server/defaults/{name} drifted from src/woollama/defaults/{name} \
             — re-copy so the Rust server and Python package ship the same defaults"
        ),
        // Not in the monorepo (e.g. a `cargo install`/published checkout) — nothing to compare.
        Err(_) => eprintln!("defaults_sync: skipping ({} absent)", py.display()),
    }
}

#[test]
fn vendored_defaults_match_python_package() {
    check("mcp.json", include_str!("../defaults/mcp.json"));
    check("recipes.toml", include_str!("../defaults/recipes.toml"));
}
