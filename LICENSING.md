# Licensing

woollama is split across crates with **two different licences**. Which one
applies depends on what you use.

| component | licence | where it ships |
|---|---|---|
| `woollama-server` — the `woollamad` router daemon | **MIT** | crates.io: [`woollama-server`](https://crates.io/crates/woollama-server) |
| `woollama-engine` — the pure-Rust engine it is built on | **MIT** | crates.io (a dependency of the server) |
| `woollama-core` — the Python-facing wheel | **AGPL-3.0-or-later** | PyPI: `woollama-core` |
| the `woollama` package at the repository root (the Python implementation) | **AGPL-3.0-or-later** | crates.io name placeholder |

The repository's top-level [`LICENSE`](LICENSE) is the AGPL text, because the
root package and `woollama-core` are AGPL. The MIT text for the two Rust crates
sits beside each of them as `LICENSE-MIT`.

## What this means in practice

- **Running or embedding `woollamad`** — the router daemon, its OpenAI server
  and its MCP server — is MIT. `woollama-server` depends on `woollama-engine`,
  which is also MIT; it does **not** link the AGPL `woollama-core`, so the
  daemon carries no copyleft obligation.
- **Using the Python package `woollama-core`** puts you under AGPL-3.0-or-later,
  including its network-use clause: if you modify it and let others use it over
  a network, you owe them the source.
- **Contributions** are taken under the licence of the crate you are editing.

If you need a combination this split doesn't cover, open an issue and say what
you're trying to do.
