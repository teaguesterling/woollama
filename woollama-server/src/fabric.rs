//! The managed/routed **fabric** backend (Part 2 of the pattern-templating plan).
//!
//! fabric is NOT OpenAI-compatible (it speaks its own REST: `POST /chat` SSE,
//! `GET /patterns/names`, `GET /patterns/{name}`, `GET /models/names`), so it cannot ride the
//! `woollama-engine` OpenAI path — it lives entirely here in the server layer. woollama either
//! **spawns + supervises** a `fabric --serve` (managed) or **routes** to an externally-run one
//! (`url`). This gives "fabric behind woollama": fabric's full library + assembly + advanced
//! features (`context`/`strategy`/`language`/`search`) without cosmic-fabric owning the process.
//!
//! Two surfaces consume this:
//!   - `/fabric/*` — a TRANSPARENT reverse-proxy of fabric's REST (no translation; the client
//!     speaks fabric natively, so vendor/SSE/advanced-features pass through verbatim).
//!   - `/w1/patterns` — fabric's library is sourced into discovery; a fabric-backed pattern
//!     run is translated to/from the OpenAI shape (see lib.rs).
//!
//! Lifecycle (managed): **reuse + graceful-kill** (mirrors cosmic-fabric's `ensure_serve`).
//! The spawned fabric is detached (NO `kill_on_drop`) and its address is persisted, so a
//! woollamad restart reuses the live fabric instead of orphaning it + re-eating fabric's
//! multi-second startup. It is killed only on graceful shutdown (main.rs). A hard crash leaves
//! at most one fabric, reclaimed next start via the readiness probe.

use std::collections::HashMap;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::Mutex;

use crate::config::FabricConfig;

pub struct FabricBackend {
    /// The fabric REST base, e.g. `http://127.0.0.1:PORT` (no trailing slash).
    pub base: String,
    client: reqwest::Client,
    /// The supervised child (managed mode only); killed on graceful shutdown.
    child: Mutex<Option<tokio::process::Child>>,
    /// `provider(lowercased) → fabric vendor name` (e.g. `ollama → "Ollama"`), derived from
    /// fabric's own `/models/names` `vendors` keys — so the mapping tracks fabric, not a guess.
    vendor_map: HashMap<String, String>,
}

/// `$XDG_RUNTIME_DIR/woollama.fabric-addr` — the persisted managed-fabric address (for reuse).
fn addr_file() -> std::path::PathBuf {
    crate::binding::runtime_dir().join("woollama.fabric-addr")
}

/// Ask the OS for an unused loopback TCP port (bind :0, read it back, drop).
fn free_port() -> Option<u16> {
    let l = std::net::TcpListener::bind("127.0.0.1:0").ok()?;
    l.local_addr().ok().map(|a| a.port())
}

impl FabricBackend {
    /// Resolve config → a live fabric (or `None` if it can't be reached/spawned). Errors are
    /// logged and degrade to `None` (the router still starts; `/fabric/*` then 503s).
    pub async fn connect(cfg: FabricConfig) -> Option<std::sync::Arc<FabricBackend>> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .ok()?;

        let (base, child) = if let Some(url) = &cfg.url {
            // External fabric — route to it, never spawn.
            let base = url.trim_end_matches('/').to_string();
            if !Self::ready(&client, &base).await {
                eprintln!("woollamad: fabric url {base} is not responding (/patterns/names) — skipping fabric backend");
                return None;
            }
            (base, None)
        } else if cfg.managed {
            match Self::ensure_serve(&client, &cfg).await {
                Some(v) => v,
                None => return None,
            }
        } else {
            eprintln!("woollamad: fabric config has neither 'url' nor 'managed: true' — skipping fabric backend");
            return None;
        };

        let vendor_map = Self::load_vendor_map(&client, &base).await;
        eprintln!("woollamad: fabric backend ready at {base} ({} vendors)", vendor_map.len());
        Some(std::sync::Arc::new(FabricBackend { base, client, child: Mutex::new(child), vendor_map }))
    }

    /// Liveness probe — fabric answers `GET /patterns/names` once `--serve` is up.
    async fn ready(client: &reqwest::Client, base: &str) -> bool {
        matches!(
            client.get(format!("{base}/patterns/names")).timeout(Duration::from_secs(2)).send().await,
            Ok(r) if r.status().is_success()
        )
    }

    /// Managed mode: reuse a live fabric at the persisted/configured address, else spawn one.
    /// Returns `(base, child)` — `child` is `None` when an existing instance was reused.
    async fn ensure_serve(
        client: &reqwest::Client,
        cfg: &FabricConfig,
    ) -> Option<(String, Option<tokio::process::Child>)> {
        let addr = Self::resolve_address(client, cfg).await?;
        let base = format!("http://{addr}");

        // Reuse a fabric already listening there (persisted-port reuse, or someone else's).
        if Self::ready(client, &base).await {
            eprintln!("woollamad: reusing live fabric at {base}");
            return Some((base, None));
        }

        // PATH: prepend ~/.local/bin (fabric's common install location). fabric is the user's
        // own tool and NEEDS provider keys to run /chat, so it inherits the full env (unlike the
        // untrusted MCP children, which are env-scrubbed).
        let mut path = std::env::var("PATH").unwrap_or_default();
        if let Ok(home) = std::env::var("HOME") {
            path = format!("{home}/.local/bin:{path}");
        }
        eprintln!("woollamad: spawning `{} --serve --address {addr}`", cfg.command);
        let mut child = tokio::process::Command::new(&cfg.command)
            .args(["--serve", "--address", &addr])
            .env("PATH", path)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            // Detached: NO kill_on_drop — survive a woollamad restart (reuse). Killed only on
            // graceful shutdown via `shutdown()`.
            .kill_on_drop(false)
            .spawn()
            .map_err(|e| eprintln!("woollamad: failed to spawn fabric: {e}"))
            .ok()?;

        // Poll readiness (~25s, like cosmic-fabric).
        for _ in 0..50 {
            if Self::ready(client, &base).await {
                eprintln!("woollamad: fabric --serve is up at {base}");
                return Some((base, Some(child)));
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        eprintln!("woollamad: fabric --serve did not come up at {base} in time");
        let _ = child.kill().await;
        None
    }

    /// Address precedence: explicit `address` config; else a persisted port from a prior run
    /// (if a fabric still answers there → reuse it); else a fresh free loopback port (persisted).
    async fn resolve_address(client: &reqwest::Client, cfg: &FabricConfig) -> Option<String> {
        if let Some(a) = &cfg.address {
            return Some(a.clone());
        }
        let pf = addr_file();
        if let Ok(prev) = std::fs::read_to_string(&pf) {
            let prev = prev.trim().to_string();
            if !prev.is_empty() && Self::ready(client, &format!("http://{prev}")).await {
                return Some(prev);
            }
        }
        let addr = format!("127.0.0.1:{}", free_port()?);
        let _ = std::fs::write(&pf, &addr);
        Some(addr)
    }

    /// Build `provider→vendor` from fabric's `/models/names` `vendors` map (keys lowercased).
    async fn load_vendor_map(client: &reqwest::Client, base: &str) -> HashMap<String, String> {
        let mut map = HashMap::new();
        if let Ok(r) = client.get(format!("{base}/models/names")).timeout(Duration::from_secs(5)).send().await {
            if let Ok(v) = r.json::<Value>().await {
                if let Some(vendors) = v.get("vendors").and_then(Value::as_object) {
                    for k in vendors.keys() {
                        map.insert(k.to_lowercase(), k.clone());
                    }
                }
            }
        }
        map
    }

    /// fabric's vendor name for a woollama provider prefix (e.g. `ollama` → `"Ollama"`). Falls
    /// back to the provider verbatim when fabric didn't enumerate it.
    pub fn vendor_for(&self, provider: &str) -> String {
        self.vendor_map.get(&provider.to_lowercase()).cloned().unwrap_or_else(|| provider.to_string())
    }

    /// fabric's pattern library (`GET /patterns/names`). Empty on error.
    pub async fn pattern_names(&self) -> Vec<String> {
        match self.client.get(format!("{}/patterns/names", self.base)).timeout(Duration::from_secs(10)).send().await {
            Ok(r) => r
                .json::<Value>()
                .await
                .ok()
                .and_then(|v| v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()))
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    /// A pattern's raw system text (`GET /patterns/{name}` → `{"Pattern": "..."}`). `None` if
    /// fabric doesn't know it.
    pub async fn pattern_system(&self, name: &str) -> Option<String> {
        // fabric pattern names are filesystem slugs (`[a-z0-9_]+`), so no percent-encoding is
        // needed here (avoids a dependency just for this one path segment).
        let r = self
            .client
            .get(format!("{}/patterns/{name}", self.base))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .ok()?;
        if !r.status().is_success() {
            return None;
        }
        let v: Value = r.json().await.ok()?;
        v.get("Pattern").and_then(Value::as_str).map(str::to_string)
    }

    /// Forward an arbitrary request to fabric (the `/fabric/*` transparent proxy). Returns the
    /// raw `reqwest::Response` so the caller can STREAM the body back (SSE-safe — never buffer).
    pub async fn forward(
        &self,
        method: reqwest::Method,
        path_and_query: &str,
        content_type: Option<&str>,
        body: bytes::Bytes,
    ) -> Result<reqwest::Response, String> {
        let url = format!("{}{}", self.base, path_and_query);
        let mut rb = self.client.request(method, &url).body(body);
        if let Some(ct) = content_type {
            rb = rb.header("content-type", ct);
        }
        rb.send().await.map_err(|e| e.to_string())
    }

    /// POST a fabric `/chat` body and return the raw streaming response (used by `/w1/run`
    /// when a pattern is fabric-backed; the caller translates fabric SSE ⇄ OpenAI).
    pub async fn chat(&self, body: &Value) -> Result<reqwest::Response, String> {
        self.client.post(format!("{}/chat", self.base)).json(body).send().await.map_err(|e| e.to_string())
    }

    /// Kill the supervised fabric (managed mode) — called on graceful shutdown only.
    pub async fn shutdown(&self) {
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
        }
    }
}
