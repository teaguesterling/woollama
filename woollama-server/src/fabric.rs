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

use std::collections::{HashMap, HashSet};
use std::process::Stdio;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::Mutex;

use crate::config::FabricConfig;

pub struct FabricBackend {
    /// The fabric REST base, e.g. `http://127.0.0.1:PORT` (no trailing slash). Immutable —
    /// managed respawns reuse the SAME address so this never changes.
    pub base: String,
    client: reqwest::Client,
    /// The supervised child (managed mode only); killed on graceful shutdown. Also the
    /// **heal lock**: `ensure_alive` holds it to single-flight respawns.
    child: Mutex<Option<tokio::process::Child>>,
    /// `provider(lowercased) → fabric vendor name` (e.g. `ollama → "Ollama"`), derived from
    /// fabric's own `/models/names` `vendors` keys — so the mapping tracks fabric, not a guess.
    vendor_map: HashMap<String, String>,
    /// fabric's pattern library. Seeded at connect, then refreshed: traffic-driven on a TTL
    /// (fabric hot-reloads its pattern dir) and after every respawn. Behind a lock so the sync
    /// `has`/`list` can read it while a background task swaps it.
    names: Arc<RwLock<HashSet<String>>>,
    /// Fallback inferencer for runs that omit `model` (fabric patterns have no bound one).
    default_model: Option<String>,
    /// Whether we supervise the child and MAY respawn it (managed mode). False in url mode.
    managed: bool,
    /// The fabric binary to (re)spawn in managed mode.
    command: String,
    /// Last time the name cache was (or began) refreshing — the TTL gate for `maybe_kick_refresh`.
    last_refresh: Arc<std::sync::Mutex<Instant>>,
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

/// The per-backend registration entry point (see `pattern_backend::register_all`): add the
/// fabric backend to the set IF mcp.json configures it. Every backend module exposes one of
/// these; the composition root calls them in order. Config/connect failures degrade to "no
/// fabric backend" (logged) rather than failing startup.
pub async fn register(backends: &mut Vec<std::sync::Arc<dyn crate::pattern_backend::PatternBackend>>) {
    match crate::config::load_fabric_config() {
        Ok(Some(cfg)) => {
            if let Some(fb) = FabricBackend::connect(cfg).await {
                backends.push(fb);
            }
        }
        Ok(None) => {}
        Err(e) => eprintln!("woollamad: fabric config error: {e}"),
    }
}

impl FabricBackend {
    /// Resolve config → a live fabric (or `None` if it can't be reached/spawned). Errors are
    /// logged and degrade to `None` (the router still starts; `/fabric/*` then 503s).
    pub async fn connect(cfg: FabricConfig) -> Option<std::sync::Arc<FabricBackend>> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .ok()?;
        // Managed = we own (and may respawn) the child. url mode routes only, never respawns.
        let is_managed = cfg.url.is_none() && cfg.managed;

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
        let names: HashSet<String> = Self::fetch_names(&client, &base).await.into_iter().collect();
        eprintln!(
            "woollamad: fabric backend ready at {base} ({} patterns, {} vendors)",
            names.len(),
            vendor_map.len()
        );
        Some(std::sync::Arc::new(FabricBackend {
            base,
            client,
            child: Mutex::new(child),
            vendor_map,
            names: Arc::new(RwLock::new(names)),
            default_model: cfg.default_model,
            managed: is_managed,
            command: cfg.command,
            last_refresh: Arc::new(std::sync::Mutex::new(Instant::now())),
        }))
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
        let mut child = Self::spawn_fabric(&cfg.command, &addr)?;
        if Self::poll_ready(client, &base, 50).await {
            eprintln!("woollamad: fabric --serve is up at {base}");
            return Some((base, Some(child)));
        }
        eprintln!("woollamad: fabric --serve did not come up at {base} in time");
        let _ = child.kill().await;
        None
    }

    /// Spawn a detached `<command> --serve --address <addr>`. fabric is the user's own tool and
    /// NEEDS provider keys to run `/chat`, so it inherits the full env (PATH gets ~/.local/bin
    /// prepended — fabric's common install location). Unlike the untrusted MCP children, which
    /// are env-scrubbed. NO `kill_on_drop`: detached so a woollamad restart can reuse it.
    fn spawn_fabric(command: &str, addr: &str) -> Option<tokio::process::Child> {
        let mut path = std::env::var("PATH").unwrap_or_default();
        if let Ok(home) = std::env::var("HOME") {
            path = format!("{home}/.local/bin:{path}");
        }
        eprintln!("woollamad: spawning `{command} --serve --address {addr}`");
        tokio::process::Command::new(command)
            .args(["--serve", "--address", addr])
            .env("PATH", path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(false)
            .spawn()
            .map_err(|e| eprintln!("woollamad: failed to spawn fabric: {e}"))
            .ok()
    }

    /// Poll `ready` up to `tries` times, 500ms apart (~25s at 50) — fabric's startup window.
    async fn poll_ready(client: &reqwest::Client, base: &str, tries: u32) -> bool {
        for _ in 0..tries {
            if Self::ready(client, base).await {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        false
    }

    /// Re-probe fabric and, in managed mode, respawn a dead/hung child on the SAME address.
    /// Single-flight via the `child` lock: concurrent callers serialize, and each re-probes
    /// first, so only the one that still finds fabric down does the kill+respawn. Returns whether
    /// fabric is live afterward. url mode (not ours) can only re-probe — it never respawns.
    async fn ensure_alive(&self) -> bool {
        let mut guard = self.child.lock().await;
        if Self::ready(&self.client, &self.base).await {
            return true; // already healthy (or a concurrent caller just healed it)
        }
        if !self.managed {
            return false; // url mode: not ours to respawn
        }
        // Reap the stale/hung child BEFORE rebinding — else the held port → "address in use".
        if let Some(mut old) = guard.take() {
            let _ = old.kill().await;
        }
        let addr = self.base.strip_prefix("http://").unwrap_or(&self.base).to_string();
        let Some(mut child) = Self::spawn_fabric(&self.command, &addr) else {
            return false;
        };
        if Self::poll_ready(&self.client, &self.base, 50).await {
            *guard = Some(child);
            // Re-source names across the restart (patterns may have been gained/lost).
            Self::refresh_names_into(&self.client, &self.base, &self.names).await;
            eprintln!("woollamad: fabric respawned at {}", self.base);
            true
        } else {
            let _ = child.kill().await;
            false
        }
    }

    /// Re-fetch fabric's pattern list and swap it into the shared cache. Never panics (runs in a
    /// detached task). An empty result (transient failure / fabric momentarily down) does NOT
    /// wipe the cache — we keep the last-known-good names.
    async fn refresh_names_into(client: &reqwest::Client, base: &str, names: &RwLock<HashSet<String>>) {
        let fresh: HashSet<String> = Self::fetch_names(client, base).await.into_iter().collect();
        if fresh.is_empty() {
            return;
        }
        if let Ok(mut w) = names.write() {
            *w = fresh;
        }
    }

    /// The name-cache TTL — default 60s, overridable via `WOOLLAMA_FABRIC_REFRESH_SECS` (0 =
    /// refresh on every read, for tests).
    fn refresh_ttl() -> Duration {
        std::env::var("WOOLLAMA_FABRIC_REFRESH_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(60))
    }

    /// Traffic-driven, single-flight name refresh. If the TTL has elapsed, claim the slot and
    /// spawn a detached re-source; otherwise a cheap no-op. fabric hot-reloads its pattern dir,
    /// so this picks up patterns gained/lost without a restart — eventually: the triggering call
    /// still sees the pre-refresh cache, the next one sees the update.
    fn maybe_kick_refresh(&self) {
        {
            let Ok(mut lr) = self.last_refresh.lock() else {
                return;
            };
            if lr.elapsed() < Self::refresh_ttl() {
                return;
            }
            *lr = Instant::now(); // claim before spawning — single-flight the kick
        }
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let client = self.client.clone();
            let base = self.base.clone();
            let names = self.names.clone();
            handle.spawn(async move {
                Self::refresh_names_into(&client, &base, &names).await;
            });
        }
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

    /// Fetch fabric's pattern library (`GET /patterns/names`). Empty on error.
    async fn fetch_names(client: &reqwest::Client, base: &str) -> Vec<String> {
        match client.get(format!("{base}/patterns/names")).timeout(Duration::from_secs(10)).send().await {
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

    /// Build fabric's `/chat` request body from a `/w1/run` body. All fabric-isms
    /// (`prompts[].userInput`/`patternName`/`vendor`, `contextName`/`strategyName`,
    /// top-level `language`/`search`) are confined here. `Err` is a client-facing 400 message.
    fn build_chat_body(&self, name: &str, body: &Value) -> Result<Value, String> {
        // Per-call `model` wins; else the configured `default_model`; else error.
        let model = match body.get("model").and_then(Value::as_str) {
            Some(m) => m,
            None => self.default_model.as_deref().ok_or_else(|| {
                format!("fabric pattern '{name}' run requires a 'model' (e.g. ollama/qwen3) — fabric patterns have no bound inferencer (set fabric.default_model in mcp.json to make it optional)")
            })?,
        };
        let (provider, bare) = model.split_once('/').unwrap_or(("", model));
        // fabric is pattern + single-input oriented: `/chat` takes ONE `userInput` string, not a
        // turn list. For an OpenAI-style messages array we concatenate every USER message's text
        // (older code kept only the LAST, silently dropping the rest). Non-user turns
        // (assistant/system) are NOT sent — fabric patterns operate on raw content, and role
        // scaffolding would change what the pattern sees. Multi-turn assistant context is lost on
        // this path by design; a client needing it should use `/fabric/*` (native fabric) directly.
        let user_input = match body.get("input") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(arr)) => arr
                .iter()
                .filter(|m| m.get("role").and_then(Value::as_str) == Some("user"))
                .filter_map(|m| m.get("content").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n"),
            _ => String::new(),
        };
        let mut prompt = serde_json::json!({
            "userInput": user_input,
            "patternName": name,
            "model": bare,
            "vendor": self.vendor_for(provider),
            "variables": body.get("variables").cloned().unwrap_or_else(|| serde_json::json!({})),
        });
        if let Some(c) = body.get("context").and_then(Value::as_str) {
            prompt["contextName"] = serde_json::json!(c);
        }
        if let Some(s) = body.get("strategy").and_then(Value::as_str) {
            prompt["strategyName"] = serde_json::json!(s);
        }
        let mut fbody = serde_json::json!({ "prompts": [prompt], "model": bare });
        if let Some(l) = body.get("language").and_then(Value::as_str) {
            fbody["language"] = serde_json::json!(l);
        }
        if let Some(opts) = body.get("options").and_then(Value::as_object) {
            for (k, v) in opts {
                fbody[k] = v.clone();
            }
        }
        if body.get("search").and_then(Value::as_bool) == Some(true) {
            fbody["search"] = serde_json::json!(true);
        }
        Ok(fbody)
    }
}

/// Parse one fabric SSE line → `(content_piece, done, error)`. fabric frames are
/// `data: {"type": "content"|"complete"|"error", "content": "..."}`. `None` = blank/unparsable.
fn parse_fabric_line(line: &str) -> Option<(Option<String>, bool, Option<String>)> {
    let line = line.trim();
    let line = line.strip_prefix("data:").map(str::trim).unwrap_or(line);
    if line.is_empty() {
        return None;
    }
    if line == "[DONE]" {
        return Some((None, true, None));
    }
    let ev: Value = serde_json::from_str(line).ok()?;
    match ev.get("type").and_then(Value::as_str) {
        Some("error") => {
            Some((None, true, Some(ev.get("content").and_then(Value::as_str).unwrap_or("fabric error").to_string())))
        }
        Some("complete") => Some((None, true, None)),
        _ => Some((ev.get("content").and_then(Value::as_str).map(String::from), false, None)),
    }
}

#[async_trait::async_trait]
impl crate::pattern_backend::PatternBackend for FabricBackend {
    fn id(&self) -> &str {
        "fabric"
    }

    fn list(&self) -> Vec<crate::pattern_backend::PatternInfo> {
        // Names only — scanning ~250 patterns' systems per discovery call is too costly;
        // variables resolve on render/run.
        self.maybe_kick_refresh();
        self.names
            .read()
            .map(|names| {
                names
                    .iter()
                    .map(|n| crate::pattern_backend::PatternInfo {
                        name: n.clone(),
                        variables: Vec::new(),
                        source: "fabric".to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn has(&self, name: &str) -> bool {
        self.maybe_kick_refresh();
        self.names.read().map(|n| n.contains(name)).unwrap_or(false)
    }

    async fn render(&self, name: &str, variables: &serde_json::Map<String, Value>) -> Option<String> {
        let system = match self.pattern_system(name).await {
            Some(s) => s,
            // Could be a genuine 404, or fabric is down — try to heal, then retry once.
            None => {
                if !self.ensure_alive().await {
                    return None;
                }
                self.pattern_system(name).await?
            }
        };
        Some(crate::config::render_system(&system, variables))
    }

    async fn run(&self, name: &str, body: &Value) -> axum::response::Response {
        use axum::response::IntoResponse;
        use futures::StreamExt;

        let fbody = match self.build_chat_body(name, body) {
            Ok(b) => b,
            Err(msg) => {
                return (
                    axum::http::StatusCode::BAD_REQUEST,
                    axum::Json(serde_json::json!({"error": {"message": msg, "type": "invalid_request_error"}})),
                )
                    .into_response()
            }
        };
        let model = body.get("model").and_then(Value::as_str).unwrap_or(name).to_string();
        let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

        // A transport error may mean fabric died — try to heal (respawn in managed mode) and
        // retry once before giving up.
        let resp = match self.chat(&fbody).await {
            Ok(r) => r,
            Err(e) => {
                let retry = if self.ensure_alive().await { self.chat(&fbody).await.ok() } else { None };
                match retry {
                    Some(r) => r,
                    None => {
                        return (
                            axum::http::StatusCode::BAD_GATEWAY,
                            axum::Json(serde_json::json!({"error": {"message": format!("fabric chat error: {e}"), "type": "server_error"}})),
                        )
                            .into_response()
                    }
                }
            }
        };
        if !resp.status().is_success() {
            let status = axum::http::StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
            let text = resp.text().await.unwrap_or_default();
            return (
                status,
                axum::Json(serde_json::json!({"error": {"message": format!("fabric: {text}"), "type": "server_error"}})),
            )
                .into_response();
        }

        if stream {
            // Translate fabric's native SSE → OpenAI chat.completion.chunk frames.
            let cid = crate::chatcmpl_id();
            let created = crate::now_secs();
            let mut byte_stream = resp.bytes_stream();
            let out = axum::body::Body::from_stream(async_stream::stream! {
                yield Ok::<bytes::Bytes, std::io::Error>(crate::chat_chunk(&cid, created, &model, serde_json::json!({"role": "assistant"}), None));
                let mut buf: Vec<u8> = Vec::new();
                'outer: while let Some(chunk) = byte_stream.next().await {
                    let Ok(bytes) = chunk else { break };
                    buf.extend_from_slice(&bytes);
                    while let Some(line) = crate::take_line(&mut buf) {
                        if let Some((piece, done, err)) = parse_fabric_line(&line) {
                            if let Some(e) = err {
                                let payload = serde_json::json!({"error": {"message": e, "type": "server_error"}});
                                yield Ok(bytes::Bytes::from(format!("data: {}\n\n", serde_json::to_string(&payload).unwrap())));
                                break 'outer;
                            }
                            if done {
                                break 'outer;
                            }
                            if let Some(c) = piece {
                                yield Ok(crate::chat_chunk(&cid, created, &model, serde_json::json!({"content": c}), None));
                            }
                        }
                    }
                }
                yield Ok(crate::chat_chunk(&cid, created, &model, serde_json::json!({}), Some("stop")));
                yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
            });
            return crate::sse_response(out);
        }

        // Non-stream: fabric still answers as SSE — accumulate content into one completion.
        let text = resp.text().await.unwrap_or_default();
        let mut content = String::new();
        let mut error = None;
        for line in text.lines() {
            if let Some((piece, done, err)) = parse_fabric_line(line) {
                if let Some(e) = err {
                    error = Some(e);
                    break;
                }
                if let Some(c) = piece {
                    content.push_str(&c);
                }
                if done {
                    break;
                }
            }
        }
        if let Some(e) = error {
            return (
                axum::http::StatusCode::BAD_GATEWAY,
                axum::Json(serde_json::json!({"error": {"message": format!("fabric: {e}"), "type": "server_error"}})),
            )
                .into_response();
        }
        axum::Json(serde_json::json!({
            "id": crate::chatcmpl_id(),
            "object": "chat.completion",
            "created": crate::now_secs(),
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        }))
        .into_response()
    }

    fn v1_addressable(&self) -> bool {
        // Only addressable as woollama/<name> in /v1 when a default model is configured (there
        // is no per-call model slot there); otherwise the pattern is /w1-only.
        self.default_model.is_some()
    }

    fn proxies(&self) -> bool {
        true
    }

    async fn proxy(
        &self,
        method: axum::http::Method,
        path_and_query: &str,
        content_type: Option<&str>,
        body: bytes::Bytes,
    ) -> axum::response::Response {
        use axum::response::IntoResponse;
        // Heal-and-retry once on a transport error (fabric may have died). Bytes clone is cheap
        // (refcounted), so the body can be replayed for the retry.
        let mut result = self.forward(method.clone(), path_and_query, content_type, body.clone()).await;
        if result.is_err() && self.ensure_alive().await {
            result = self.forward(method, path_and_query, content_type, body).await;
        }
        match result {
            Ok(resp) => crate::pattern_backend::stream_reqwest(resp),
            Err(e) => (
                axum::http::StatusCode::BAD_GATEWAY,
                axum::Json(serde_json::json!({"error": {"message": format!("fabric proxy error: {e}"), "type": "server_error"}})),
            )
                .into_response(),
        }
    }

    async fn shutdown(&self) {
        FabricBackend::shutdown(self).await;
    }
}
