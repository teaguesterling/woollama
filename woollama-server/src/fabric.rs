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
        eprintln!("woollamad: spawning `{command} --serve --address {addr}`");
        tokio::process::Command::new(command)
            .args(["--serve", "--address", addr])
            .env("PATH", fabric_path_env())
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
        let model = self.resolve_model(body, name)?;
        let (provider, bare) = model.split_once('/').unwrap_or(("", model));
        // fabric is pattern + single-input oriented: `/chat` takes ONE `userInput` string, not a
        // turn list. `extract_user_input` concatenates every USER message's text (older code kept
        // only the LAST, silently dropping the rest, and dropped array-`content` text entirely).
        // Non-user turns (assistant/system) are NOT sent — fabric patterns operate on raw content,
        // and role scaffolding would change what the pattern sees. Multi-turn assistant context is
        // lost on this path by design; a client needing it should use `/fabric/*` directly. Any
        // images are ignored here — the REST path has no attachment field (vision takes the CLI).
        let (user_input, _images, _bad) = extract_user_input(body.get("input"));
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

    /// Resolve the inferencer for a fabric run: per-call `model` wins; else the configured
    /// `default_model`; else a 400-worthy error (fabric patterns have no bound inferencer).
    fn resolve_model<'a>(&'a self, body: &'a Value, name: &str) -> Result<&'a str, String> {
        match body.get("model").and_then(Value::as_str) {
            Some(m) => Ok(m),
            None => self.default_model.as_deref().ok_or_else(|| {
                format!("fabric pattern '{name}' run requires a 'model' (e.g. ollama/qwen3) — fabric patterns have no bound inferencer (set fabric.default_model in mcp.json to make it optional)")
            }),
        }
    }

    /// Run a fabric pattern with image input via the one-shot CLI (`fabric -a`), since fabric's
    /// REST `/chat` has no attachment field. `user_text`/`images` are pre-extracted from the body;
    /// `images` is guaranteed non-empty by the caller. The text is piped on stdin; the first image
    /// becomes `--attachment=` (URL passed through, data-URL written to a temp file cleaned up on
    /// every exit path). Plain-text stdout → an OpenAI completion (or the SSE *shape* if the caller
    /// asked to stream). NOTE: needs a VISION-capable `model` — a text model (e.g. the usual
    /// `default_model`) will see no image. Returns an OpenAI-shaped error response on any failure.
    async fn run_fabric_vision(
        &self,
        name: &str,
        body: &Value,
        user_text: String,
        mut images: Vec<ImageRef>,
    ) -> axum::response::Response {
        use axum::response::IntoResponse;
        use axum::http::StatusCode;

        let model = match self.resolve_model(body, name) {
            Ok(m) => m.to_string(),
            Err(msg) => return err_resp(StatusCode::BAD_REQUEST, msg, "invalid_request_error"),
        };
        let (provider, bare) = model.split_once('/').unwrap_or(("", model.as_str()));
        let vendor = self.vendor_for(provider);

        // fabric's `-a` is single-attachment: use the first image, note any dropped extras.
        let extra = images.len().saturating_sub(1);
        let first = images.remove(0); // caller guarantees non-empty
        if extra > 0 {
            eprintln!("woollamad: fabric vision: {extra} extra image(s) ignored (`-a` is single-attachment)");
        }

        // Materialize the attachment. Keep the temp file alive (binding `_tmp`) until after the
        // subprocess exits — dropping the `NamedTempFile` unlinks it, so cleanup is automatic on
        // EVERY return below it (success, non-zero exit, or wait error).
        let (attachment, _tmp) = match first {
            ImageRef::Url(u) => (u, None),
            ImageRef::Data { bytes, ext } => {
                let tmp = tempfile::Builder::new().prefix("woollama-vision-").suffix(&format!(".{ext}")).tempfile();
                match tmp {
                    Ok(mut f) => {
                        use std::io::Write;
                        if let Err(e) = f.as_file_mut().write_all(&bytes) {
                            return err_resp(StatusCode::BAD_GATEWAY, format!("vision temp write failed: {e}"), "server_error");
                        }
                        (f.path().to_string_lossy().into_owned(), Some(f))
                    }
                    Err(e) => return err_resp(StatusCode::BAD_GATEWAY, format!("vision temp file failed: {e}"), "server_error"),
                }
            }
        };

        let variables = body.get("variables").and_then(Value::as_object).cloned().unwrap_or_default();
        let argv = build_fabric_argv(name, bare, &vendor, &attachment, &variables);

        // One-shot CLI, env-inherited (needs PATH→~/.local/bin + provider keys, like spawn_fabric).
        let mut child = match tokio::process::Command::new(&self.command)
            .args(&argv)
            .env("PATH", fabric_path_env())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                return err_resp(StatusCode::BAD_GATEWAY, format!("fabric CLI spawn failed (`{}`): {e}", self.command), "server_error")
            }
        };
        // Write stdin on a SEPARATE task and drain stdout via `wait_with_output` CONCURRENTLY.
        // Doing them in sequence (write-all-then-wait) deadlocks if the child emits enough stdout
        // to fill the pipe buffer before it finishes reading stdin: the child blocks on write while
        // we block on write. Concurrent write + drain is the only safe shape.
        if let Some(mut sin) = child.stdin.take() {
            tokio::spawn(async move {
                use tokio::io::AsyncWriteExt;
                let _ = sin.write_all(user_text.as_bytes()).await;
                let _ = sin.shutdown().await; // close stdin so fabric stops reading and proceeds
            });
        }
        let out = match child.wait_with_output().await {
            Ok(o) => o,
            Err(e) => return err_resp(StatusCode::BAD_GATEWAY, format!("fabric CLI wait failed: {e}"), "server_error"),
        };
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return err_resp(StatusCode::BAD_GATEWAY, format!("fabric CLI exited {}: {}", out.status, stderr.trim()), "server_error");
        }
        let content = String::from_utf8_lossy(&out.stdout).into_owned();

        if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
            // The CLI is one-shot, so there's nothing to stream incrementally — but a client that
            // asked for `stream:true` still gets the OpenAI SSE SHAPE (role, one content chunk,
            // stop, [DONE]) rather than a surprise JSON body.
            let cid = crate::chatcmpl_id();
            let created = crate::now_secs();
            let model_s = model.clone();
            let out_stream = axum::body::Body::from_stream(async_stream::stream! {
                yield Ok::<bytes::Bytes, std::io::Error>(crate::chat_chunk(&cid, created, &model_s, serde_json::json!({"role": "assistant"}), None));
                yield Ok(crate::chat_chunk(&cid, created, &model_s, serde_json::json!({"content": content}), None));
                yield Ok(crate::chat_chunk(&cid, created, &model_s, serde_json::json!({}), Some("stop")));
                yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
            });
            return crate::sse_response(out_stream);
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

// --- Vision via the fabric CLI (`fabric -a`) --------------------------------------------------
// fabric's REST `/chat` has NO attachment field, so image input (OpenAI `image_url` content
// parts) can't ride the REST path — it takes fabric's one-shot CLI instead:
// `fabric --pattern=<name> --attachment=<path|url> --model=<m> --vendor=<V>`, userInput piped on
// stdin, plain-text answer on stdout (ground-truthed against llama3.2-vision: stdin in, text out,
// exit 0). Non-streaming; a `stream:true` request still gets the OpenAI SSE *shape* back.

/// Cap on a decoded `data:` image (20 MiB) — defense against a giant base64 payload OOM-ing the
/// process or filling the temp dir. Larger than any real photo; http(s) images have no local
/// decode so they're not bound by this.
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// An image referenced by an OpenAI `image_url` content part.
enum ImageRef {
    /// `http(s)://…` — passed to fabric's `-a` directly (fabric fetches it).
    Url(String),
    /// A decoded `data:<mime>;base64,…` payload — written to a temp file for `-a`.
    Data { bytes: Vec<u8>, ext: String },
}

/// Map an image MIME type to a temp-file extension (fabric infers attachment type from the path).
/// Unknown types fall back to `img` — the vendor still sniffs the bytes.
fn mime_to_ext(mime: &str) -> &'static str {
    match mime.trim().to_ascii_lowercase().as_str() {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/bmp" => "bmp",
        _ => "img",
    }
}

/// Parse one OpenAI `image_url.url` into an [`ImageRef`]. Accepts `http(s)://` URLs (passed
/// through) and `data:<mime>;base64,<payload>` data-URLs (decoded). Returns `None` for anything
/// else — notably a bare filesystem path, which a request must NOT be able to hand to the CLI.
fn parse_image_url(url: &str) -> Option<ImageRef> {
    use base64::Engine;
    let u = url.trim();
    if u.starts_with("http://") || u.starts_with("https://") {
        return Some(ImageRef::Url(u.to_string()));
    }
    // `data:<mime>;base64,<payload>` — only base64 data-URLs are supported.
    let rest = u.strip_prefix("data:")?;
    let (meta, payload) = rest.split_once(',')?;
    let meta = meta.strip_suffix(";base64")?; // require base64 encoding
    let mime = meta.split(';').next().unwrap_or(""); // drop any extra params
    let payload = payload.trim();
    // Reject an over-cap payload BEFORE decoding (base64 decodes to ~3/4 its length), so we never
    // allocate the huge buffer just to throw it away.
    if payload.len() / 4 * 3 > MAX_IMAGE_BYTES {
        return None;
    }
    // Accept both padded (STANDARD) and unpadded payloads — clients vary.
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload))
        .ok()?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return None;
    }
    Some(ImageRef::Data { bytes, ext: mime_to_ext(mime).to_string() })
}

/// Append one OpenAI message `content` (a string, or an array of `{type:text|image_url}` parts)
/// to the running text + image lists. `bad_images` counts `image_url` parts that were PRESENT but
/// couldn't be turned into an [`ImageRef`] (undecodable data-URL, or a rejected non-URL) — so the
/// caller can tell "no image" apart from "image we couldn't use" and fail loudly instead of
/// silently answering text-only.
fn collect_content(content: &Value, texts: &mut Vec<String>, images: &mut Vec<ImageRef>, bad_images: &mut usize) {
    match content {
        Value::String(s) => texts.push(s.clone()),
        Value::Array(parts) => {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            texts.push(t.to_string());
                        }
                    }
                    Some("image_url") => {
                        let url = part.get("image_url").and_then(|i| i.get("url")).and_then(Value::as_str);
                        match url.and_then(parse_image_url) {
                            Some(img) => images.push(img),
                            None => *bad_images += 1,
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Pull the user text and any image attachments out of a `/w1/run` `input` (a bare string, or an
/// OpenAI messages array whose `content` is a string or an array of parts). Text is `\n\n`-joined
/// across user turns/parts; images keep request order. Used by BOTH the REST and vision paths, so
/// array-content TEXT is no longer silently dropped (it was — when `content` was an array,
/// `as_str()` returned `None`, losing both text and images).
fn extract_user_input(input: Option<&Value>) -> (String, Vec<ImageRef>, usize) {
    let mut texts: Vec<String> = Vec::new();
    let mut images: Vec<ImageRef> = Vec::new();
    let mut bad_images = 0usize;
    match input {
        Some(Value::String(s)) => texts.push(s.clone()),
        Some(Value::Array(msgs)) => {
            for m in msgs {
                if m.get("role").and_then(Value::as_str) == Some("user") {
                    if let Some(c) = m.get("content") {
                        collect_content(c, &mut texts, &mut images, &mut bad_images);
                    }
                }
            }
        }
        _ => {}
    }
    (texts.join("\n\n"), images, bad_images)
}

/// Assemble the fabric CLI argv for a vision run. Every value uses the `--flag=value` form (NOT
/// `--flag value`), so a value beginning with `-` can't be reparsed as a flag; the userInput is
/// piped on stdin, never an argv. Variables use fabric's `-v=#key:value` syntax.
fn build_fabric_argv(
    pattern: &str,
    model_bare: &str,
    vendor: &str,
    attachment: &str,
    variables: &serde_json::Map<String, Value>,
) -> Vec<String> {
    let mut argv = vec![
        format!("--pattern={pattern}"),
        format!("--model={model_bare}"),
        format!("--vendor={vendor}"),
        format!("--attachment={attachment}"),
    ];
    for (k, v) in variables {
        let val = match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        argv.push(format!("--variable=#{k}:{val}"));
    }
    argv
}

/// PATH with `~/.local/bin` prepended — fabric's common install location, needed by both the
/// `--serve` spawn and the one-shot vision CLI.
fn fabric_path_env() -> String {
    let mut path = std::env::var("PATH").unwrap_or_default();
    if let Ok(home) = std::env::var("HOME") {
        path = format!("{home}/.local/bin:{path}");
    }
    path
}

/// An OpenAI-shaped error response (matches the inline errors the REST `run` path returns).
fn err_resp(status: axum::http::StatusCode, msg: String, typ: &str) -> axum::response::Response {
    use axum::response::IntoResponse;
    (status, axum::Json(serde_json::json!({"error": {"message": msg, "type": typ}}))).into_response()
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

        // Image input can't ride fabric's REST `/chat` (no attachment field) — route it to the
        // one-shot `fabric -a` CLI. Text-only requests stay on the REST path below.
        let (user_text, images, bad_images) = extract_user_input(body.get("input"));
        if !images.is_empty() {
            return self.run_fabric_vision(name, body, user_text, images).await;
        }
        // The request DID carry image_url part(s) but none were usable — fail loudly rather than
        // silently answering text-only (which would look like the image was seen and ignored).
        if bad_images > 0 {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                "image_url could not be used — expected an http(s) URL or a `data:<mime>;base64,…` URL".to_string(),
                "invalid_request_error",
            );
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use serde_json::json;

    fn data_url(mime: &str, bytes: &[u8]) -> String {
        format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    #[test]
    fn mime_to_ext_maps_known_and_falls_back() {
        assert_eq!(mime_to_ext("image/png"), "png");
        assert_eq!(mime_to_ext("image/jpeg"), "jpg");
        assert_eq!(mime_to_ext("IMAGE/WEBP"), "webp"); // case-insensitive
        assert_eq!(mime_to_ext("application/octet-stream"), "img"); // unknown → fallback
    }

    #[test]
    fn parse_image_url_handles_http_data_and_rejects_paths() {
        // http(s) → passed through verbatim.
        match parse_image_url("https://example.com/cat.png") {
            Some(ImageRef::Url(u)) => assert_eq!(u, "https://example.com/cat.png"),
            _ => panic!("expected Url"),
        }
        // data-URL → decoded bytes + extension from the MIME type.
        match parse_image_url(&data_url("image/png", b"\x89PNG\r\n")) {
            Some(ImageRef::Data { bytes, ext }) => {
                assert_eq!(bytes, b"\x89PNG\r\n");
                assert_eq!(ext, "png");
            }
            _ => panic!("expected Data"),
        }
        // A bare filesystem path must NOT be accepted (a request can't hand the CLI a local file).
        assert!(parse_image_url("/etc/passwd").is_none());
        assert!(parse_image_url("file:///etc/passwd").is_none());
        // Non-base64 data-URL (no `;base64`) is rejected.
        assert!(parse_image_url("data:image/png,notbase64").is_none());
    }

    #[test]
    fn parse_image_url_rejects_over_cap_data_url() {
        // A base64 payload whose decoded size would exceed MAX_IMAGE_BYTES is rejected up front
        // (no giant allocation). 'A' is a valid base64 char; length a multiple of 4 needs no pad.
        let over = MAX_IMAGE_BYTES / 3 * 4 + 8;
        let big = format!("data:image/png;base64,{}", "A".repeat(over));
        assert!(parse_image_url(&big).is_none(), "over-cap image rejected");
        // Just under the cap still parses.
        let under = format!("data:image/png;base64,{}", "A".repeat(1024));
        assert!(matches!(parse_image_url(&under), Some(ImageRef::Data { .. })));
    }

    #[test]
    fn parse_image_url_accepts_unpadded_base64() {
        // Same 3 bytes, padded vs unpadded — both must decode.
        assert!(matches!(parse_image_url("data:image/png;base64,YWJj"), Some(ImageRef::Data { .. })));
        match parse_image_url("data:image/png;base64,YWJj") {
            Some(ImageRef::Data { bytes, .. }) => assert_eq!(bytes, b"abc"),
            _ => panic!("expected Data"),
        }
    }

    #[test]
    fn extract_user_input_bare_string() {
        let (text, imgs, bad) = extract_user_input(Some(&json!("hello world")));
        assert_eq!(text, "hello world");
        assert!(imgs.is_empty());
        assert_eq!(bad, 0);
    }

    #[test]
    fn extract_user_input_counts_unusable_images() {
        // An `image_url` part that can't be parsed (bare path, undecodable data-URL) is counted,
        // so `run()` can 400 instead of silently answering text-only.
        let input = json!([{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "/etc/passwd"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%notb64%%%"}}
        ]}]);
        let (text, imgs, bad) = extract_user_input(Some(&input));
        assert_eq!(text, "hi");
        assert!(imgs.is_empty(), "neither image is usable");
        assert_eq!(bad, 2, "both unusable image_url parts counted");
    }

    #[test]
    fn extract_user_input_array_content_keeps_text_and_images() {
        // The latent-bug case: `content` is an ARRAY of parts. Old code dropped it entirely
        // (`as_str()` → None); now text is kept AND the image is extracted.
        let input = json!([
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": data_url("image/jpeg", b"JPGDATA")}}
            ]},
            {"role": "user", "content": "and this?"}
        ]);
        let (text, imgs, _bad) = extract_user_input(Some(&input));
        assert_eq!(text, "what is this?\n\nand this?", "text joined across parts/turns, system dropped");
        assert_eq!(imgs.len(), 1);
        match &imgs[0] {
            ImageRef::Data { bytes, ext } => {
                assert_eq!(bytes, b"JPGDATA");
                assert_eq!(ext, "jpg");
            }
            _ => panic!("expected decoded Data image"),
        }
    }

    #[test]
    fn extract_user_input_preserves_multiple_image_order() {
        let input = json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://a/1.png"}},
            {"type": "image_url", "image_url": {"url": "https://b/2.png"}}
        ]}]);
        let (_t, imgs, _bad) = extract_user_input(Some(&input));
        assert_eq!(imgs.len(), 2);
        match (&imgs[0], &imgs[1]) {
            (ImageRef::Url(a), ImageRef::Url(b)) => {
                assert_eq!(a, "https://a/1.png");
                assert_eq!(b, "https://b/2.png");
            }
            _ => panic!("expected two Url images in order"),
        }
    }

    #[test]
    fn build_fabric_argv_uses_equals_form_and_fabric_var_syntax() {
        let mut vars = serde_json::Map::new();
        vars.insert("role".into(), json!("expert"));
        let argv = build_fabric_argv("summarize", "llama3.2-vision:latest", "Ollama", "/tmp/x.png", &vars);
        assert!(argv.contains(&"--pattern=summarize".to_string()));
        assert!(argv.contains(&"--model=llama3.2-vision:latest".to_string()));
        assert!(argv.contains(&"--vendor=Ollama".to_string()));
        assert!(argv.contains(&"--attachment=/tmp/x.png".to_string()));
        assert!(argv.contains(&"--variable=#role:expert".to_string()));
        // Every value is bound with `=` (never a separate argv token), so a value beginning with
        // `-` can't be reparsed as a flag.
        let dashy = build_fabric_argv("p", "m", "V", "--evil=1", &serde_json::Map::new());
        assert!(dashy.iter().all(|a| a.starts_with("--")));
        assert!(dashy.contains(&"--attachment=--evil=1".to_string()));
    }
}
