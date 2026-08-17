//! The woollama router service (Rust) — the axum HTTP surface over `woollama-engine`.
//!
//! Surface (slices 2–8, see docs/rust-router-port.md):
//!   - `GET /v1/models` — inferencer discovery (static + live) + recipes (slice 8).
//!   - `POST /v1/chat/completions` — passthrough (`<provider>/<model>`, incl. native
//!     num_ctx + streaming) and `woollama/<recipe>` orchestration (incl. streaming),
//!     dispatching tools to the downstream MCP registry; claude-code recipes execute
//!     via the claude CLI.
//!   - `POST /v1/responses` — stateless (incl. streaming) and STATEFUL (claude-resume,
//!     store-backed, managed-agents) with the requires_action pause/resume.
//!   - `/v1/conversations` CRUD + `/items` — the durable handle table.
//!   - `/mcp` — woollama AS an MCP server (Streamable-HTTP), plus a `mcp` stdio subcommand.
//!
//! Remaining: the Unix-socket surface; the cutover (slice 9). Managed-agents' Anthropic
//! wire format is best-effort pending live reconciliation (see managed_agents.rs).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value};

use woollama_engine as engine;
use engine::EngineError;

pub mod auth;
pub mod binding;
mod claude_code;
mod config;
mod conversations;
mod fabric;
mod managed_agents;
mod mcp_registry;
mod mcp_surface;
mod ollama_native;
mod pattern_backend;
pub mod pool;
mod responses;

use mcp_surface::WoollamaMcp;
use pattern_backend::PatternBackend;

pub use config::{load_mcp_servers, load_recipes};
// The reconnect surface is public so an integration test can drive it, and because per-server
// health is meant to be surfaced rather than kept internal.
pub use config::{HttpSpec, McpServerSpec, StdioSpec};
pub use mcp_registry::{spawn_reconnect, McpRegistry, ServerHealth, ServerStatus};

/// `woollamad check-config` — validate the config files and report, without connecting to
/// anything or binding a port. Returns the process exit code: 0 clean, 1 if anything is wrong.
///
/// This exists because a malformed `mcp.json` entry is *skipped*, not fatal (one server's typo
/// must not cost an operator the other eleven), and a skip is otherwise announced only in the
/// boot log — one journal line, read approximately never. A connection failure may self-heal on
/// the next request; a config error will still be there in six weeks. So strictness lives in a
/// deliberate step the operator runs before a reload, rather than in the daemon's startup path.
/// Config faults that must STOP the daemon rather than degrade it.
///
/// `build_state` deliberately degrades most problems — a downstream that won't start is skipped so
/// one bad server can't cost an operator the other eleven. That is right for a *world* fault and
/// wrong for a *config* fault: a `mcp.json` referencing a variable that does not exist makes the
/// whole file unusable, and degrading it means starting with ZERO MCP servers, `conversationStore`
/// silently back to `None` (statelessness restored for non-claude models), no fabric backend —
/// while reporting healthy. Three stderr lines nobody re-reads, which is precisely what
/// `check-config` exists to prevent.
///
/// So this runs BEFORE `build_state` and refuses. It matches the Python reference, which raises
/// out of its lifespan and does not start.
pub fn fatal_config_error() -> Option<String> {
    config::ensure_examples_dir();
    if let Err(e) = config::diagnose_mcp_servers() {
        return Some(e);
    }
    if let Err(e) = config::load_capabilities() {
        return Some(e);
    }
    if let Err(e) = engine::Registry::from_config() {
        return Some(e.message);
    }
    None
}

pub fn check_config() -> i32 {
    config::ensure_examples_dir();
    let mut errors = 0usize;

    match config::diagnose_mcp_servers() {
        Err(e) => {
            eprintln!("error: {e}");
            errors += 1;
        }
        Ok((specs, per_server, warnings)) => {
            for e in &per_server {
                eprintln!("error: {e}");
            }
            for w in &warnings {
                eprintln!("warning: {w}");
            }
            errors += per_server.len();
            let mut names: Vec<&str> = specs.keys().map(String::as_str).collect();
            names.sort_unstable();
            println!(
                "mcp.json: {} server(s) usable{}{}",
                specs.len(),
                if names.is_empty() { String::new() } else { format!(" ({})", names.join(", ")) },
                if per_server.is_empty() { String::new() } else { format!(", {} skipped", per_server.len()) },
            );
        }
    }

    match config::load_recipes() {
        Err(e) => {
            eprintln!("error: recipes.toml: {e}");
            errors += 1;
        }
        Ok(r) => println!("recipes.toml: {} recipe(s)", r.len()),
    }

    match engine::Registry::from_config() {
        Err(e) => {
            eprintln!("error: {e}");
            errors += 1;
        }
        Ok(_) => println!("inferencers.toml: OK"),
    }

    if errors == 0 {
        println!("config OK");
        0
    } else {
        eprintln!("{errors} problem(s) found");
        1
    }
}

/// Shared, process-lifetime server state: loaded recipes, the connected downstream MCP
/// registry, and the inferencer registry. Built once at startup, shared via axum state.
pub struct AppState {
    pub recipes: HashMap<String, config::Recipe>,
    pub registry: Arc<mcp_registry::McpRegistry>,
    pub inferencers: engine::Registry,
    /// One `(DeviceModelManager, Gate)` pair per management-capable inferencer
    /// (declares a `management_url`), built from `inferencers` at startup. Consulted
    /// by the chat passthrough to take the pooled (load-on-demand + queued) path.
    pub pools: Arc<pool::PoolRegistry>,
    /// The mcp.json specs (for claude-code delegation, which writes a per-recipe
    /// --mcp-config from the referenced subset).
    pub mcp_specs: HashMap<String, config::McpServerSpec>,
    /// Per-inferencer `[inferencers.<name>.capabilities]` declarations (issue #20). Empty for an
    /// inferencer that declares none, which means "unknown" everywhere and so no behaviour change.
    pub capabilities: HashMap<String, config::CapabilityMap>,
    /// The durable conversation handle table (stateful /v1/responses + /v1/conversations).
    pub conversations: Arc<conversations::Conversations>,
    /// An external conversation store (issue #2), wired from mcp.json's
    /// `conversationStore`. When present, non-claude models become stateful (store-backed).
    pub store: Option<Arc<dyn conversations::StoreProvider>>,
    /// The Anthropic Managed Agents backend (claude-agent/* models). Paid; errors at
    /// turn time if ANTHROPIC_API_KEY is unset.
    pub managed_agents: Arc<managed_agents::ManagedAgents>,
    /// Pluggable `/w1/` pattern backends (fabric, future providers), constructed from config.
    /// The native recipes path is the built-in core; these are consulted after it (recipes win
    /// on a name collision, then registration order). See `pattern_backend`.
    pub pattern_backends: Vec<Arc<dyn PatternBackend>>,
}

impl AppState {
    fn backend_for_model(&self, model: &str) -> Option<&'static str> {
        conversations::backend_for_model(model, self.store.is_some())
    }
}

/// Load config + connect the downstream MCP servers. Errors are logged and degraded to
/// empty (the router still starts) rather than fatal.
pub async fn build_state() -> AppState {
    // Resolve WOOLLAMA_EXAMPLES_DIR before any config load — the bundled mcp.json expands it.
    config::ensure_examples_dir();
    let mut recipes = config::load_recipes().unwrap_or_else(|e| {
        eprintln!("woollamad: recipes load error: {e}");
        HashMap::new()
    });
    // Opt-in `[patterns]` directory scan (fabric-style patterns). recipes.toml wins on a
    // name collision — a hand-authored recipe overrides an auto-discovered pattern.
    match config::load_patterns() {
        Ok(patterns) => {
            for (name, r) in patterns {
                recipes.entry(name).or_insert(r);
            }
        }
        Err(e) => eprintln!("woollamad: patterns load error: {e}"),
    }
    let capabilities = config::load_capabilities().unwrap_or_else(|e| {
        eprintln!("woollamad: capabilities load error: {e}");
        HashMap::new()
    });
    let specs = config::load_mcp_servers().unwrap_or_else(|e| {
        eprintln!("woollamad: mcp.json load error: {e}");
        HashMap::new()
    });
    let registry = Arc::new(mcp_registry::McpRegistry::connect(specs.clone()).await);
    // Retry anything that didn't come up. Background-only: a request never triggers a fetch, so
    // `list_tools` keeps serving a cached snapshot and federated topologies can't recurse.
    mcp_registry::spawn_reconnect(registry.clone(), specs.clone());
    let inferencers = engine::Registry::from_config().unwrap_or_else(|e| {
        eprintln!("woollamad: inferencers load error: {e}");
        engine::Registry::new()
    });
    let management_protocols = engine::load_management_protocols().unwrap_or_else(|e| {
        eprintln!("woollamad: management_protocols load error: {e}");
        HashMap::new()
    });
    // `from_registry` never fails the whole registry on a bad `management_protocol`
    // name — it warns and skips just the offending inferencer (see its doc comment) —
    // so there's no error path here to degrade-to-empty from.
    let pools = Arc::new(pool::PoolRegistry::from_registry(&inferencers, &management_protocols));
    // Durable handle table at $WOOLLAMA_STATE_DIR/conversations.json (in-memory if unset).
    let state_path = std::env::var("WOOLLAMA_STATE_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|d| std::path::PathBuf::from(d).join("conversations.json"));
    let conversations = Arc::new(conversations::Conversations::new(state_path));
    // Optional external conversation store (makes non-claude models stateful).
    let store: Option<Arc<dyn conversations::StoreProvider>> = match config::load_conversation_store() {
        Ok(Some(config::ConvStoreConfig::Http { url })) => Some(Arc::new(conversations::HttpStoreProvider::new(&url))),
        Ok(Some(config::ConvStoreConfig::Mcp { server })) => {
            Some(Arc::new(conversations::McpStoreProvider::new(registry.clone(), server)))
        }
        Ok(None) => None,
        Err(e) => {
            eprintln!("woollamad: conversationStore config error: {e}");
            None
        }
    };
    // Opt-in: persist + reuse the managed-agents env/agent ids across restarts (default off →
    // each process creates its own). Requires WOOLLAMA_STATE_DIR for somewhere to write them.
    let ma_persist = {
        let on = std::env::var("WOOLLAMA_MANAGED_AGENTS_PERSIST")
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        let dir = std::env::var("WOOLLAMA_STATE_DIR").ok().filter(|s| !s.is_empty());
        match (on, dir) {
            (true, Some(d)) => Some(std::path::PathBuf::from(d).join("managed_agents.json")),
            (true, None) => {
                eprintln!("woollamad: WOOLLAMA_MANAGED_AGENTS_PERSIST is set but WOOLLAMA_STATE_DIR is unset \u{2014} env/agent ids will NOT persist");
                None
            }
            _ => None,
        }
    };
    let managed_agents = Arc::new(managed_agents::ManagedAgents::new(ma_persist));
    // Pluggable pattern backends, assembled from config by the composition root — this is the
    // ONLY place backends are constructed, and it names no concrete backend type (see
    // `pattern_backend::register_all`). Registration order = dispatch order after native recipes.
    let pattern_backends = pattern_backend::register_all().await;
    AppState {
        recipes,
        registry,
        inferencers,
        pools,
        mcp_specs: specs,
        capabilities,
        conversations,
        store,
        managed_agents,
        pattern_backends,
    }
}

/// The TCP host/port to bind — `$WOOLLAMA_ADDRESS=host[:port]`, else `127.0.0.1:0`.
pub fn resolve_tcp_target() -> (String, u16) {
    match std::env::var("WOOLLAMA_ADDRESS") {
        Ok(addr) if !addr.is_empty() => parse_tcp_address(&addr),
        _ => ("127.0.0.1".to_string(), 0),
    }
}

/// Parse a `WOOLLAMA_ADDRESS` value into `(host, port)`. Handles IPv4/host `host:port`,
/// bracketed IPv6 `[::1]:8080`, a bare IP with no port (`127.0.0.1`, `::1` → ephemeral port),
/// `:port` (empty host → loopback), and a bare host. Port defaults to 0 (ephemeral) when
/// absent/unparseable.
///
/// The old first-`:`-split broke every IPv6 form: a bare `::1` silently became `127.0.0.1`, and
/// `[::]:8080`/`[::1]:8080` produced host `"["` — which then PANICS `TcpListener::bind`. (The auth
/// bind-gate fails safe on the bad host, but a legitimate IPv6 bind was simply non-functional.)
fn parse_tcp_address(addr: &str) -> (String, u16) {
    // A full socket address, including bracketed IPv6 with a port (`[::1]:8080`).
    if let Ok(sa) = addr.parse::<std::net::SocketAddr>() {
        return (sa.ip().to_string(), sa.port());
    }
    // A bare IP with no port (IPv4 or IPv6, optionally bracketed) → ephemeral port.
    let bare = addr.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(addr);
    if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
        return (ip.to_string(), 0);
    }
    // Hostname forms: split on the LAST `:` so a bare hostname stays whole and `host:port` /
    // `:port` work. (Anything bracketed or multi-colon already matched a branch above.)
    match addr.rsplit_once(':') {
        Some((host, port)) => (
            if host.is_empty() { "127.0.0.1".to_string() } else { host.to_string() },
            port.parse().unwrap_or(0),
        ),
        None => (addr.to_string(), 0),
    }
}

/// `GET /v1/tools` — what tools this router actually has, and the health of every configured
/// downstream (issue #23).
///
/// Two things make this more than a debugging nicety once federation is in play:
/// a downstream that is retrying appears here **with its last error**, rather than being silently
/// absent — absence and not-yet-connected are indistinguishable from the outside, and a router
/// that showed neither would look healthy with its tools quietly gone. And each tool names the
/// server it came from, which is the only way to read a federated namespace without driving an
/// MCP handshake by hand.
///
/// `tools` keeps the Python reference's shape (`<server>.<tool>` names); `data` and `servers` are
/// additive.
async fn list_tools(State(state): State<Arc<AppState>>) -> Response {
    // ONE snapshot for both halves: two reads would let a reconnect land between them and return
    // a server reported `retrying` with 0 tools whose tools already appear in `data`.
    let (listing, statuses) = state.registry.introspect();
    let tools: Vec<String> = listing.iter().map(|(s, bare, _)| format!("{s}.{bare}")).collect();
    let data: Vec<Value> = listing
        .iter()
        .map(|(server, bare, wire)| json!({"name": wire, "server": server, "tool": bare}))
        .collect();
    let servers: Vec<Value> = statuses
        .into_iter()
        .map(|s| {
            let mut o = json!({
                "name": s.name,
                "transport": s.transport,
                "health": s.health.as_str(),
                "tools": s.tools,
            });
            match &s.health {
                crate::mcp_registry::ServerHealth::Retrying { attempts, last_error } => {
                    o["attempts"] = json!(attempts);
                    o["last_error"] = json!(last_error);
                }
                crate::mcp_registry::ServerHealth::Failed { reason } => o["reason"] = json!(reason),
                crate::mcp_registry::ServerHealth::Connected => {}
            }
            o
        })
        .collect();
    axum::Json(json!({"tools": tools, "data": data, "servers": servers})).into_response()
}

/// The axum app (shared by the binary and the integration tests). Mounts woollama's own
/// MCP surface at `/mcp` (Streamable-HTTP) on the same port — the per-session factory
/// shares the one `AppState` (and thus the one downstream registry).
pub fn router(state: Arc<AppState>) -> Router {
    use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use rmcp::transport::streamable_http_server::StreamableHttpService;

    let mcp_state = state.clone();
    let mcp_svc = StreamableHttpService::new(
        move || Ok(WoollamaMcp { state: mcp_state.clone() }),
        Arc::new(LocalSessionManager::default()),
        Default::default(),
    );
    let mut router = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/tools", get(list_tools))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/images/generations", post(images_generations))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/responses", post(responses_create))
        .route("/v1/conversations", post(conversations_create).get(conversations_list))
        .route("/v1/conversations/{conv_id}", get(conversations_get).delete(conversations_delete))
        .route("/v1/conversations/{conv_id}/items", get(conversations_items))
        .route("/w1/patterns", get(w1_patterns))
        .route("/w1/patterns/{name}/render", post(w1_render))
        .route("/w1/patterns/{name}/run", post(w1_run));
    // Mount a transparent reverse-proxy at `/{id}/*` for each backend that offers one — the id
    // comes from `backend.id()`, so no backend name is hardcoded here. Reserved prefixes are
    // skipped so a backend can't shadow woollama's own surface.
    const RESERVED: &[&str] = &["v1", "w1", "mcp"];
    for b in &state.pattern_backends {
        let id = b.id();
        if !b.proxies() {
            continue;
        }
        if RESERVED.contains(&id) {
            eprintln!("woollamad: backend '{id}' proxy NOT mounted — reserved path prefix");
            continue;
        }
        router = router
            .route(&format!("/{id}"), any(backend_proxy))
            .route(&format!("/{id}/"), any(backend_proxy))
            .route(&format!("/{id}/{{*rest}}"), any(backend_proxy));
    }
    // Raise the request-body cap from axum's 2 MiB default: base64-encoded vision images
    // (`image_url` data-URLs on `/w1/.../run` and `/v1/chat/completions`) routinely exceed it, so
    // a real photo would 413 before reaching the model. 32 MiB comfortably holds a max-size image
    // (`fabric::MAX_IMAGE_BYTES` = 20 MiB decoded ≈ 27 MiB base64) plus the JSON envelope.
    router
        .nest_service("/mcp", mcp_svc)
        .layer(axum::extract::DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}

/// Serve woollama's MCP surface over stdio — the `woollamad mcp` subcommand (what
/// an MCP client puts in its mcp.json). stdout is the JSON-RPC channel; logs go to stderr.
pub async fn serve_mcp_stdio(state: Arc<AppState>) -> Result<(), Box<dyn std::error::Error>> {
    let running = rmcp::serve_server(WoollamaMcp { state }, (tokio::io::stdin(), tokio::io::stdout())).await?;
    running.waiting().await?;
    Ok(())
}

// --- error helpers ------------------------------------------------------------

fn err_response(status: StatusCode, message: impl Into<String>, kind: &str) -> Response {
    (status, Json(json!({"error": {"message": message.into(), "type": kind}}))).into_response()
}

/// 503 + `Retry-After: <secs>` for a `pool::PoolError::Backpressure` — the one error
/// shape `err_response`/`engine_err_response` can't produce (they have no way to set
/// a header). Mirrors Python's `resp.headers["Retry-After"] = str(int(e.retry_after))`.
fn backpressure_response(retry_after_secs: f64) -> Response {
    Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header("retry-after", (retry_after_secs as u64).to_string())
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"error": {"message": "model busy; retry shortly", "type": "server_error"}}).to_string(),
        ))
        .unwrap_or_else(|_| err_response(StatusCode::SERVICE_UNAVAILABLE, "model busy; retry shortly", "server_error"))
}

fn engine_err_response(e: EngineError) -> Response {
    let status = StatusCode::from_u16(e.status.unwrap_or(500) as u16)
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let kind = e.kind.clone().unwrap_or_else(|| "server_error".to_string());
    match e.payload {
        Some(payload) => (status, Json(payload)).into_response(),
        None => err_response(status, e.message, &kind),
    }
}

async fn forward_post(
    url: String,
    body: &Value,
    headers: &HashMap<String, String>,
    timeout: u64,
) -> Result<reqwest::Response, Response> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout))
        .build()
        .map_err(|e| err_response(StatusCode::BAD_GATEWAY, e.to_string(), "server_error"))?;
    let mut rb = client.post(url).json(body);
    for (k, v) in headers {
        rb = rb.header(k, v);
    }
    rb.send()
        .await
        .map_err(|e| err_response(StatusCode::BAD_GATEWAY, e.to_string(), "server_error"))
}

async fn relay_json(resp: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let data: Value = resp.json().await.unwrap_or_else(|_| json!({}));
    (status, Json(data)).into_response()
}

// --- SSE helpers --------------------------------------------------------------

pub(crate) fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

pub(crate) fn chatcmpl_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4().simple())
}

/// Next complete `\n`-terminated line from a raw byte buffer (UTF-8-safe), or None.
pub(crate) fn take_line(buf: &mut Vec<u8>) -> Option<String> {
    let nl = buf.iter().position(|&b| b == b'\n')?;
    let line: Vec<u8> = buf.drain(..=nl).collect();
    Some(String::from_utf8_lossy(&line).into_owned())
}

/// One OpenAI `chat.completion.chunk` SSE frame.
pub(crate) fn chat_chunk(cid: &str, created: i64, model: &str, delta: Value, finish: Option<&str>) -> Bytes {
    let payload = json!({
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    });
    Bytes::from(format!("data: {}\n\n", serde_json::to_string(&payload).unwrap()))
}

pub(crate) fn sse_response(body: Body) -> Response {
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(body)
        .unwrap_or_else(|_| err_response(StatusCode::BAD_GATEWAY, "stream build failed", "server_error"))
}

/// num_ctx + stream → native `/api/chat` NDJSON, translated frame-by-frame to OpenAI SSE.
async fn native_stream(
    inf: &engine::Inferencer,
    fwd: &Value,
    headers: &HashMap<String, String>,
    model: &str,
) -> Response {
    let url = ollama_native::native_chat_url(&inf.base_url);
    let req = ollama_native::to_native_request(fwd);
    let resp = match forward_post(url, &req, headers, 600).await {
        Ok(r) => r,
        Err(e) => return e,
    };
    if resp.status().as_u16() >= 400 {
        return relay_json(resp).await;
    }
    let model = model.to_string();
    let body = Body::from_stream(stream! {
        let mut t = ollama_native::SseTranslator::new(&model);
        let mut buf: Vec<u8> = Vec::new();
        let mut bs = resp.bytes_stream();
        while let Some(chunk) = bs.next().await {
            let Ok(bytes) = chunk else { break };
            buf.extend_from_slice(&bytes);
            while let Some(line) = take_line(&mut buf) {
                for out in t.translate(&line) {
                    yield Ok::<Bytes, std::io::Error>(Bytes::from(out));
                }
            }
        }
    });
    sse_response(body)
}

/// woollama/<recipe> + stream → run the loop over SSE and emit `chat.completion.chunk`
/// frames: a role chunk, the content deltas, then exactly one stop terminator + [DONE].
/// Primed before returning so a setup/first-turn error maps to an HTTP status.
async fn orchestrate_stream(
    state: Arc<AppState>,
    recipe: config::Recipe,
    messages: Value,
    model: String,
) -> Response {
    // claude-code is non-streaming: run it, then surface the whole answer as one delta.
    if let Some(cc_model) = recipe.inferencer.strip_prefix("claude-code/") {
        let text = match run_claude_recipe(&state, &recipe, &messages, cc_model).await {
            Ok(resp) => resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string(),
            Err(e) => return engine_err_response(e),
        };
        let cid = chatcmpl_id();
        let created = now_secs();
        let body = Body::from_stream(stream! {
            yield Ok::<Bytes, std::io::Error>(chat_chunk(&cid, created, &model, json!({"role": "assistant"}), None));
            if !text.is_empty() {
                yield Ok(chat_chunk(&cid, created, &model, json!({"content": text}), None));
            }
            yield Ok(chat_chunk(&cid, created, &model, json!({}), Some("stop")));
            yield Ok(Bytes::from("data: [DONE]\n\n"));
        });
        return sse_response(body);
    }
    let recipe_val = recipe.to_value();
    let provider: Arc<dyn engine::ToolProvider> =
        Arc::new(mcp_registry::RegistryToolProvider { reg: state.registry.clone() });
    let setup = match engine::build_setup(&recipe_val, &messages, provider, None, None, Some(&state.inferencers)) {
        Ok(s) => s,
        Err(e) => return engine_err_response(e),
    };
    let mut s = Box::pin(engine::events_stream(setup, true));
    let first_ev = match s.next().await {
        Some(Err(e)) => return engine_err_response(e),
        Some(Ok(ev)) => Some(ev),
        None => None,
    };
    let cid = chatcmpl_id();
    let created = now_secs();
    let body = Body::from_stream(stream! {
        yield Ok::<Bytes, std::io::Error>(chat_chunk(&cid, created, &model, json!({"role": "assistant"}), None));
        if let Some(engine::Event::Delta(c)) = first_ev {
            yield Ok(chat_chunk(&cid, created, &model, json!({"content": c}), None));
        }
        while let Some(item) = s.next().await {
            match item {
                Ok(engine::Event::Delta(c)) => {
                    yield Ok(chat_chunk(&cid, created, &model, json!({"content": c}), None));
                }
                Ok(_) => {}
                Err(e) => {
                    let payload = e.payload.clone().unwrap_or_else(|| json!({"error": {"message": e.message, "type": e.kind}}));
                    yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&payload).unwrap())));
                    break;
                }
            }
        }
        yield Ok(chat_chunk(&cid, created, &model, json!({}), Some("stop")));
        yield Ok(Bytes::from("data: [DONE]\n\n"));
    });
    sse_response(body)
}

/// Stream a stateless /v1/responses turn as OpenAI Responses SSE (the canonical
/// created → output_item.added → content_part.added → output_text.delta* →
/// output_text.done → content_part.done → output_item.done → completed sequence).
/// Primed before returning so a setup error maps to an HTTP status.
async fn responses_stream(mut source: BoxStream<'static, Result<String, EngineError>>, model: String) -> Response {
    let first_delta = match source.next().await {
        Some(Err(e)) => return engine_err_response(e),
        Some(Ok(d)) => Some(d),
        None => None,
    };
    let resp_id = responses::new_id("resp");
    let item_id = responses::new_id("msg");
    let created = now_secs();
    let body = Body::from_stream(stream! {
        let mut seq = 0i64;
        yield Ok::<Bytes, std::io::Error>(responses::resp_ev("response.created", seq,
            json!({"response": responses::build_response_full(&resp_id, &model, "", "in_progress", created)}))); seq += 1;
        yield Ok(responses::resp_ev("response.output_item.added", seq,
            json!({"output_index": 0, "item": responses::msg_item(&item_id, "", false)}))); seq += 1;
        yield Ok(responses::resp_ev("response.content_part.added", seq,
            json!({"item_id": item_id, "output_index": 0, "content_index": 0,
                   "part": {"type": "output_text", "text": "", "annotations": []}}))); seq += 1;

        let mut chunks: Vec<String> = Vec::new();
        if let Some(d) = first_delta {
            chunks.push(d.clone());
            yield Ok(responses::resp_ev("response.output_text.delta", seq,
                json!({"item_id": item_id, "output_index": 0, "content_index": 0, "logprobs": [], "delta": d}))); seq += 1;
        }
        while let Some(item) = source.next().await {
            match item {
                Ok(piece) => {
                    chunks.push(piece.clone());
                    yield Ok(responses::resp_ev("response.output_text.delta", seq,
                        json!({"item_id": item_id, "output_index": 0, "content_index": 0, "logprobs": [], "delta": piece}))); seq += 1;
                }
                Err(e) => {
                    yield Ok(responses::resp_ev("error", seq, json!({"message": e.message, "code": e.kind}))); seq += 1;
                    break;
                }
            }
        }
        let full = chunks.concat();
        yield Ok(responses::resp_ev("response.output_text.done", seq,
            json!({"item_id": item_id, "output_index": 0, "content_index": 0, "logprobs": [], "text": full}))); seq += 1;
        yield Ok(responses::resp_ev("response.content_part.done", seq,
            json!({"item_id": item_id, "output_index": 0, "content_index": 0,
                   "part": {"type": "output_text", "text": full, "annotations": []}}))); seq += 1;
        yield Ok(responses::resp_ev("response.output_item.done", seq,
            json!({"output_index": 0, "item": responses::msg_item(&item_id, &full, true)}))); seq += 1;
        yield Ok(responses::resp_ev("response.completed", seq,
            json!({"response": responses::build_response_full(&resp_id, &model, &full, "completed", created)})));
    });
    sse_response(body)
}

// --- orchestration (shared by chat-completions + responses) -------------------

/// The mcp.json entry for each server a recipe's tools reference — the subset claude-code
/// delegation hands the child as its `--mcp-config`. Errors if a referenced server isn't
/// configured.
///
/// `env` is forwarded so a server behaves identically in woollama's own loop and under
/// delegation; omitting it would make a tool work one way and silently misbehave the other.
fn referenced_mcp_servers(
    specs: &HashMap<String, config::McpServerSpec>,
    tools: &[String],
) -> Result<HashMap<String, Value>, EngineError> {
    let mut servers = HashMap::new();
    for t in tools {
        let server = t.split_once('.').map(|(s, _)| s).unwrap_or(t.as_str());
        if servers.contains_key(server) {
            continue;
        }
        let Some(spec) = specs.get(server) else {
            return Err(EngineError::new(
                format!("recipe references MCP server '{server}' not in mcp.json config"),
                "invalid_request_error",
                400,
            ));
        };
        let entry = match spec {
            config::McpServerSpec::Stdio(s) => {
                json!({"command": s.command, "args": s.args, "env": s.env})
            }
            // Refused rather than translated. Claude Code's mcp.json CAN express an HTTP server,
            // but emitting one would have the child connect to the downstream ITSELF — a network
            // peer woollama never brokers — putting it outside the allow-list boundary that makes
            // delegation containable. (Secrets on disk are not the distinction: the stdio arm
            // above already writes `env` into the same file, which `claude_code` creates under a
            // 0700 `tempfile::tempdir()` that unlinks on drop.) Out of scope for issue #19.
            config::McpServerSpec::Http(_) => {
                return Err(EngineError::new(
                    format!(
                        "recipe references MCP server '{server}', which is a 'url' (HTTP) server — \
                         claude-code delegation cannot hand an HTTP downstream to the child process"
                    ),
                    "invalid_request_error",
                    400,
                ))
            }
        };
        servers.insert(server.to_string(), entry);
    }
    Ok(servers)
}

/// Run a `claude-code/<model>` recipe: tool-less completion, or delegation when the
/// recipe allow-lists tools (Claude owns the loop). Returns an OpenAI dict.
async fn run_claude_recipe(
    state: &AppState,
    recipe: &config::Recipe,
    messages: &Value,
    model: &str,
) -> Result<Value, EngineError> {
    let cc_err = |e: claude_code::ClaudeCodeError| EngineError::new(format!("claude-code backend: {e}"), "server_error", 502);
    if recipe.tools.is_empty() {
        claude_code::run_completion(&recipe.system, messages, model).await.map_err(cc_err)
    } else {
        let servers = referenced_mcp_servers(&state.mcp_specs, &recipe.tools)?;
        claude_code::run_delegated(&recipe.system, messages, model, &recipe.tools, &servers, 8)
            .await
            .map_err(cc_err)
    }
}

/// Run a recipe to completion and return the final OpenAI response dict. A
/// `claude-code/<model>` recipe runs through the executor; otherwise tools are
/// dispatched to the downstream MCP registry via the engine loop. Shared by the HTTP
/// handlers and the MCP `chat` tool; each maps the `EngineError` to its own surface.
pub(crate) async fn orchestrate_recipe(
    state: &AppState,
    recipe: &config::Recipe,
    messages: &Value,
) -> Result<Value, EngineError> {
    if let Some(cc_model) = recipe.inferencer.strip_prefix("claude-code/") {
        return run_claude_recipe(state, recipe, messages, cc_model).await;
    }
    let recipe_val = recipe.to_value();
    let provider: Arc<dyn engine::ToolProvider> =
        Arc::new(mcp_registry::RegistryToolProvider { reg: state.registry.clone() });
    let setup = engine::build_setup(&recipe_val, messages, provider, None, None, Some(&state.inferencers))?;
    let mut s = Box::pin(engine::events_stream(setup, false));
    while let Some(item) = s.next().await {
        match item {
            Ok(engine::Event::Final(resp)) => return Ok(resp),
            Ok(_) => continue,
            Err(e) => return Err(e),
        }
    }
    Err(EngineError::new("orchestrate: loop ended without a final response", "server_error", 500))
}

// --- GET /v1/models -----------------------------------------------------------

/// `GET /v1/models` (slice 8): each inferencer's opted-in models (static `models` +
/// optional live `discover`, namespaced `provider/<id>`) plus `woollama/<recipe>`.
async fn list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut data: Vec<Value> = Vec::new();
    for inf in state.inferencers.list() {
        let mut seen = std::collections::HashSet::new();
        let mut ids = inf.models.clone();
        if inf.discover {
            if let Ok(found) = discover_models(&inf).await {
                ids.extend(found);
            }
        }
        let caps = state.capabilities.get(&inf.name).cloned().unwrap_or_default();
        for id in ids {
            if seen.insert(id.clone()) {
                let mut entry =
                    json!({"id": format!("{}/{id}", inf.name), "object": "model", "owned_by": inf.name});
                // Only DECLARED capability is surfaced here: this list includes models that are not
                // resident, and discovery only describes what is loaded. Absent means undeclared,
                // never "cannot".
                let declared: Vec<&str> = caps
                    .iter()
                    .filter(|(_, pats)| pats.iter().any(|p| fnmatch(p, &id)))
                    .map(|(name, _)| name.as_str())
                    .collect();
                if !declared.is_empty() {
                    entry["capabilities"] = json!(declared);
                }
                data.push(entry);
            }
        }
    }
    let mut recipe_names: Vec<String> = state.recipes.keys().cloned().collect();
    recipe_names.sort();
    for r in recipe_names {
        data.push(json!({"id": format!("woollama/{r}"), "object": "model", "owned_by": "woollama"}));
    }
    // Backend-sourced patterns are addressable as `woollama/<name>` too — but only when the
    // backend can actually run them here (no per-call model slot in /v1), so /v1/models stays
    // honest. (They're always in /w1/patterns.) Minus names already claimed.
    let mut seen: std::collections::HashSet<String> = state.recipes.keys().cloned().collect();
    for backend in &state.pattern_backends {
        if !backend.v1_addressable() {
            continue;
        }
        let mut infos = backend.list();
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        for info in infos {
            if seen.insert(info.name.clone()) {
                data.push(json!({"id": format!("woollama/{}", info.name), "object": "model", "owned_by": "woollama"}));
            }
        }
    }
    Json(json!({"object": "list", "data": data}))
}

/// Live-query a provider's own `/v1/models`, filtered by its `model_patterns` (empty =
/// all). Errors (missing key / unreachable / non-200) are the caller's cue to skip.
async fn discover_models(inf: &engine::Inferencer) -> Result<Vec<String>, ()> {
    let headers = inf.auth_headers().map_err(|_| ())?;
    let client = reqwest::Client::builder().timeout(Duration::from_secs(10)).build().map_err(|_| ())?;
    let mut rb = client.get(format!("{}/models", inf.base_url.trim_end_matches('/')));
    for (k, v) in &headers {
        rb = rb.header(k, v);
    }
    let r = rb.send().await.map_err(|_| ())?;
    if !r.status().is_success() {
        return Err(());
    }
    let v: Value = r.json().await.map_err(|_| ())?;
    let mut ids: Vec<String> = v
        .get("data")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|m| m.get("id").and_then(Value::as_str).map(String::from)).collect())
        .unwrap_or_default();
    if !inf.model_patterns.is_empty() {
        ids.retain(|id| inf.model_patterns.iter().any(|p| fnmatch(p, id)));
    }
    Ok(ids)
}

/// fnmatch-style glob (`*` any run, `?` one char) — mirrors Python's `fnmatch` filtering.
pub(crate) fn fnmatch(pattern: &str, name: &str) -> bool {
    fn m(p: &[u8], n: &[u8]) -> bool {
        match p.split_first() {
            None => n.is_empty(),
            Some((b'*', rest)) => m(rest, n) || (!n.is_empty() && m(p, &n[1..])),
            Some((b'?', rest)) => !n.is_empty() && m(rest, &n[1..]),
            Some((c, rest)) => !n.is_empty() && n[0] == *c && m(rest, &n[1..]),
        }
    }
    m(pattern.as_bytes(), name.as_bytes())
}

// --- /w1/ (woollama-native): pattern templating -------------------------------
// `/v1/*` is OpenAI-compatible; templating is not an OpenAI concept, so it lives under
// woollama's own `/w1/` namespace. Patterns ARE recipes (see config::Recipe::render):
// render substitutes `{{var}}`, then the EXISTING orchestration path runs. Patterns also
// stay in `/v1/models` as `woollama/<name>` for OpenAI-client addressability.

/// The `variables` array a native recipe surfaces in `/w1/patterns`: one object per scanned
/// `{{var}}` (in [`config::scan_vars`] order), enriched with the recipe's
/// [`config::Recipe::variables`] overlay (`default`/`choices`/`description`). Absent metadata
/// fields are omitted, so a plain `{{var}}` with no overlay is just `{"name": "x"}`. The list
/// is driven by `scan_vars` (what the template actually uses); an overlay entry whose name
/// isn't in the system is dropped (we never advertise a variable that substitutes nothing).
fn w1_variable_infos(recipe: &config::Recipe) -> Vec<Value> {
    config::scan_vars(&recipe.system)
        .into_iter()
        .map(|name| {
            let mut obj = serde_json::Map::new();
            if let Some(meta) = recipe.variables.get(&name) {
                if let Some(default) = &meta.default {
                    obj.insert("default".into(), default.clone());
                }
                if let Some(choices) = &meta.choices {
                    obj.insert("choices".into(), Value::Array(choices.clone()));
                }
                if let Some(description) = &meta.description {
                    obj.insert("description".into(), json!(description));
                }
            }
            obj.insert("name".into(), json!(name));
            Value::Object(obj)
        })
        .collect()
}

/// `GET /w1/patterns` — discovery. Native recipes (with scanned `{{var}}` names + any
/// metadata overlay) + fabric's library (source `"fabric"`, names only). On a name collision
/// the native recipe WINS (recipes.toml/dir scan override an auto-sourced fabric pattern), so
/// it is the one listed.
async fn w1_patterns(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut entries: Vec<(&String, &config::Recipe)> = state.recipes.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut data: Vec<Value> = entries
        .iter()
        .map(|(name, r)| {
            json!({"name": name, "variables": w1_variable_infos(r), "source": r.source.as_str()})
        })
        .collect();
    // Pluggable backends, after native (native recipes win on a name collision, then
    // registration order). De-dup so two backends offering the same name list it once.
    let mut seen: std::collections::HashSet<String> = state.recipes.keys().cloned().collect();
    for backend in &state.pattern_backends {
        let mut infos = backend.list();
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        for info in infos {
            if seen.insert(info.name.clone()) {
                data.push(json!({"name": info.name, "variables": info.variables, "source": info.source}));
            }
        }
    }
    Json(json!({"data": data}))
}

/// `POST /w1/patterns/{name}/render` — render-without-run (cosmic-fabric's `assemble`).
/// Body `{input, variables}` → `{"prompt": "<system, {{vars}} substituted>\n\n<input>"}`.
async fn w1_render(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let variables = body.get("variables").and_then(Value::as_object).cloned().unwrap_or_default();
    let input = body.get("input").and_then(Value::as_str).unwrap_or("");
    // Native recipe WINS on a name collision; else the first backend that has it.
    let system = if let Some(recipe) = state.recipes.get(&name) {
        // Fill author-configured defaults for any variable the caller didn't supply.
        let variables = recipe.apply_defaults(&variables);
        Some(config::render_system(&recipe.system, &variables))
    } else {
        let mut rendered = None;
        for backend in &state.pattern_backends {
            if backend.has(&name) {
                rendered = backend.render(&name, &variables).await;
                break;
            }
        }
        rendered
    };
    match system {
        Some(system) => Json(json!({"prompt": format!("{system}\n\n{input}")})).into_response(),
        None => err_response(StatusCode::NOT_FOUND, format!("unknown pattern '{name}'"), "not_found"),
    }
}

/// `POST /w1/patterns/{name}/run` — templated run + infer. Body `{input (string | OpenAI
/// messages array), variables, model (per-call inferencer override), stream, options}`.
/// Renders the pattern, then reuses the EXISTING orchestration/streaming path → an OpenAI
/// chat-completion object (or OpenAI SSE when `stream:true`).
async fn w1_run(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    // Native recipe WINS on a name collision; else dispatch to the first backend that has it.
    if let Some(recipe) = state.recipes.get(&name) {
        return w1_run_native(&state, recipe, &name, &body).await;
    }
    for backend in &state.pattern_backends {
        if backend.has(&name) {
            return backend.run(&name, &body).await;
        }
    }
    err_response(StatusCode::NOT_FOUND, format!("unknown pattern '{name}'"), "not_found")
}

/// Run a NATIVE recipe pattern through the engine path (render `{{var}}`, per-call model +
/// options overrides, then the existing orchestration/streaming dispatch).
async fn w1_run_native(state: &Arc<AppState>, recipe: &config::Recipe, name: &str, body: &Value) -> Response {
    let variables = body.get("variables").and_then(Value::as_object).cloned().unwrap_or_default();
    // Fill author-configured defaults for any variable the caller didn't supply (the same
    // overlay `/w1/patterns/{name}/render` applies — both route through `apply_defaults`).
    let variables = recipe.apply_defaults(&variables);
    let model_override = body.get("model").and_then(Value::as_str);
    let mut rendered = recipe.render(&variables, model_override);
    // Per-call `options` (e.g. temperature) override the recipe's bound params.
    if let Some(opts) = body.get("options").and_then(Value::as_object) {
        let mut params = rendered.params.as_ref().and_then(Value::as_object).cloned().unwrap_or_default();
        for (k, v) in opts {
            params.insert(k.clone(), v.clone());
        }
        rendered.params = Some(Value::Object(params));
    }
    // `input` is either a bare string (→ one user message) or an OpenAI messages array.
    let messages = match body.get("input") {
        Some(Value::Array(arr)) => Value::Array(arr.clone()),
        Some(Value::String(s)) => json!([{"role": "user", "content": s}]),
        _ => json!([]),
    };
    let model_label = model_override.map(String::from).unwrap_or_else(|| format!("woollama/{name}"));
    if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        return orchestrate_stream(state.clone(), rendered, messages, model_label).await;
    }
    match orchestrate_recipe(state, &rendered, &messages).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => engine_err_response(e),
    }
}

/// `/{backend-id}/*` — TRANSPARENT reverse-proxy of a pattern backend's native API. The
/// backend (selected by the first path segment) streams the response back verbatim — SSE-safe,
/// status + content-type preserved. Backends carry provider keys, so woollama's bind stays
/// loopback/UDS by default. Generic: no backend name appears here.
async fn backend_proxy(
    State(state): State<Arc<AppState>>,
    method: axum::http::Method,
    uri: axum::extract::OriginalUri,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Response {
    let full = uri.0.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let id = full.trim_start_matches('/').split(['/', '?']).next().unwrap_or("");
    let Some(backend) = state.pattern_backends.iter().find(|b| b.id() == id && b.proxies()) else {
        return err_response(StatusCode::SERVICE_UNAVAILABLE, format!("no backend '{id}' configured"), "server_error");
    };
    // Strip the `/{id}` mount prefix, preserving the rest of the path + query string.
    let rest = full.strip_prefix(&format!("/{id}")).filter(|s| !s.is_empty()).unwrap_or("/");
    let ct = headers.get("content-type").and_then(|v| v.to_str().ok());
    backend.proxy(method, rest, ct, body).await
}

// --- POST /v1/chat/completions ------------------------------------------------

async fn chat_completions(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();

    if let Some(name) = model.strip_prefix("woollama/") {
        // Native recipe WINS; else a backend pattern (it resolves its own model — fabric uses
        // fabric.default_model since /v1 has no per-call model slot).
        if let Some(recipe) = state.recipes.get(name) {
            let messages = body.get("messages").cloned().unwrap_or_else(|| json!([]));
            if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
                return orchestrate_stream(state.clone(), recipe.clone(), messages, model).await;
            }
            return match orchestrate_recipe(&state, recipe, &messages).await {
                Ok(resp) => Json(resp).into_response(),
                Err(e) => engine_err_response(e),
            };
        }
        for backend in &state.pattern_backends {
            if backend.has(name) {
                // Re-shape the OpenAI request as a `/w1/run`-style body (messages as input);
                // no `model` — the backend supplies it (e.g. fabric.default_model).
                let run_body = json!({
                    "input": body.get("messages").cloned().unwrap_or_else(|| json!([])),
                    "stream": body.get("stream").and_then(Value::as_bool).unwrap_or(false),
                });
                return backend.run(name, &run_body).await;
            }
        }
        return err_response(StatusCode::NOT_FOUND, format!("unknown recipe '{name}'"), "not_found");
    }

    let provider = model.split('/').next().unwrap_or("");
    let Some(inf) = state.inferencers.resolve(provider) else {
        return err_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unknown model namespace: '{model}'. Use 'woollama/<recipe>' or \
                 '<provider>/<model>' for a known inferencer ({}).",
                state.inferencers.names().join(", ")
            ),
            "invalid_request_error",
        );
    };

    let bare = model.split_once('/').map_or("", |(_, rest)| rest).to_string();

    // Refuse a CONCRETE model the operator declared cannot chat, before either dispatch path.
    // Deliberately above the pooled/non-pooled split: a pooled inferencer is not the only kind
    // that can be pointed at an embedder, and a check that only guards one path is worse than
    // none because it reads as covered. (`default` is not a concrete id — it is resolved against
    // residency and filtered there.)
    let caps = state.capabilities.get(provider).cloned().unwrap_or_default();
    if bare != "default" {
        if let Some(r) = reject_wrong_capability(&caps, &bare, "chat", provider) {
            return r;
        }
    }

    // Management-capable inferencer (declares a `management_url`) with a built pool:
    // resolve virtual models, load-on-demand, and queue/serialize through the Gate.
    // Everything else (incl. a management_url inferencer somehow missing its pool)
    // keeps today's exact stateless-relay path, unchanged.
    if inf.management_url.is_some() {
        if let Some((manager, gate)) = state.pools.get(provider) {
            return passthrough_pooled(&inf, &caps, manager, gate, &body, &bare).await;
        }
    }

    let headers = match inf.auth_headers() {
        Ok(h) => h,
        Err(e) => return engine_err_response(e),
    };

    let mut fwd = body.clone();
    fwd["model"] = json!(bare);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    if provider == "ollama" && ollama_native::wants_native(&fwd) {
        if stream {
            return native_stream(&inf, &fwd, &headers, &bare).await;
        }
        return passthrough_native(&inf, &fwd, &headers, &bare).await;
    }

    if stream {
        return passthrough_stream(&inf, &fwd, &headers).await;
    }

    fwd["stream"] = json!(false);
    match forward_post(inf.chat_url(), &fwd, &headers, 180).await {
        Ok(resp) => relay_json(resp).await,
        Err(e) => e,
    }
}

// --- POST /v1/images/generations ----------------------------------------------

/// Text-to-image passthrough: `<provider>/<model>` -> that inferencer's OpenAI-compat
/// `/v1/images/generations` (e.g. the device's Z-Image-Turbo). Always non-streaming. Image
/// generation runs for tens of seconds, so it gets a generous timeout rather than the chat
/// path's 180s.
async fn images_generations(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let provider = model.split('/').next().unwrap_or("");
    let Some(inf) = state.inferencers.resolve(provider) else {
        return err_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unknown model namespace: '{model}'. Use '<provider>/<model>' for a known \
                 inferencer ({}).",
                state.inferencers.names().join(", ")
            ),
            "invalid_request_error",
        );
    };
    // Refuse a model the operator has declared cannot serve this endpoint, rather than letting
    // the backend answer — issue #20's motivating case is a backend that responds to an
    // unsupported request by taking the whole model service down.
    let bare = model.split_once('/').map(|(_, b)| b).unwrap_or("");
    let caps = state.capabilities.get(provider).cloned().unwrap_or_default();
    if let Some(r) = reject_wrong_capability(&caps, bare, "image", provider) {
        return r;
    }

    let headers = match inf.auth_headers() {
        Ok(h) => h,
        Err(e) => return engine_err_response(e),
    };

    let bare = model.split_once('/').map_or("", |(_, rest)| rest).to_string();
    let mut fwd = body.clone();
    fwd["model"] = json!(bare);

    match forward_post(inf.images_url(), &fwd, &headers, 300).await {
        Ok(resp) => relay_json(resp).await,
        Err(e) => e,
    }
}

// --- POST /v1/embeddings -------------------------------------------------------

/// Text-embedding passthrough: `<provider>/<model>` -> that inferencer's OpenAI-compat
/// `/v1/embeddings` (e.g. the device's Qwen3-Embedding). For local vectorization/RAG through
/// woollama. Embeddings are quick, so the chat path's 180s timeout is plenty.
async fn embeddings(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let provider = model.split('/').next().unwrap_or("");
    let Some(inf) = state.inferencers.resolve(provider) else {
        return err_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unknown model namespace: '{model}'. Use '<provider>/<model>' for a known \
                 inferencer ({}).",
                state.inferencers.names().join(", ")
            ),
            "invalid_request_error",
        );
    };
    // Refuse a model the operator has declared cannot serve this endpoint, rather than letting
    // the backend answer — issue #20's motivating case is a backend that responds to an
    // unsupported request by taking the whole model service down.
    let bare = model.split_once('/').map(|(_, b)| b).unwrap_or("");
    let caps = state.capabilities.get(provider).cloned().unwrap_or_default();
    if let Some(r) = reject_wrong_capability(&caps, bare, "embedding", provider) {
        return r;
    }

    let headers = match inf.auth_headers() {
        Ok(h) => h,
        Err(e) => return engine_err_response(e),
    };

    let bare = model.split_once('/').map_or("", |(_, rest)| rest).to_string();
    let mut fwd = body.clone();
    fwd["model"] = json!(bare);

    match forward_post(inf.embeddings_url(), &fwd, &headers, 180).await {
        Ok(resp) => relay_json(resp).await,
        Err(e) => e,
    }
}

async fn passthrough_native(
    inf: &engine::Inferencer,
    fwd: &Value,
    headers: &HashMap<String, String>,
    model: &str,
) -> Response {
    let url = ollama_native::native_chat_url(&inf.base_url);
    let req = ollama_native::to_native_request(fwd);
    let resp = match forward_post(url, &req, headers, 600).await {
        Ok(r) => r,
        Err(e) => return e,
    };
    if resp.status().as_u16() >= 400 {
        return relay_json(resp).await;
    }
    match resp.json::<Value>().await {
        Ok(native) => Json(ollama_native::from_native_response(&native, model)).into_response(),
        Err(e) => err_response(StatusCode::BAD_GATEWAY, e.to_string(), "server_error"),
    }
}

async fn passthrough_stream(
    inf: &engine::Inferencer,
    fwd: &Value,
    headers: &HashMap<String, String>,
) -> Response {
    let resp = match forward_post(inf.chat_url(), fwd, headers, 180).await {
        Ok(r) => r,
        Err(e) => return e,
    };
    if resp.status().as_u16() >= 400 {
        return relay_json(resp).await;
    }
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(resp.bytes_stream()))
        .unwrap_or_else(|_| err_response(StatusCode::BAD_GATEWAY, "stream build failed", "server_error"))
}

/// Warn once per inferencer that `default` resolution has fallen open. Once, not per request:
/// this sits on the request path, and the condition persists until either the config or the
/// device's residency changes.
fn warn_fail_open(name: &str) {
    static WARNED: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
    let mut guard = WARNED.lock().expect("warn set poisoned");
    let seen = guard.get_or_insert_with(std::collections::HashSet::new);
    if seen.insert(name.to_string()) {
        eprintln!(
            "woollamad: inferencer '{name}': none of its configured `models` is loaded on the \
             device, so `{name}/default` is falling back to ANY resident model — which may be one \
             that cannot serve this endpoint (an embedding or rerank model will fail the chat \
             path). Add the model you expect to serve `default` to this inferencer's `models`."
        );
    }
}

/// Refuse a request routed to a model the operator has positively declared cannot serve this
/// endpoint. `None` = allowed.
///
/// Only *declared* capability is consulted here, not discovered: discovery describes what is
/// RESIDENT, and this check runs for concrete model ids that may not be loaded at all. The
/// point is to turn a class of backend failure into a clear 400 — issue #20's motivating bug is
/// an unsupported request taking a whole model service down, where the backend's own answer is a
/// 500 and every later request fails until reload. Unknown stays allowed, so nothing regresses.
fn reject_wrong_capability(
    caps: &config::CapabilityMap,
    model: &str,
    capability: &str,
    provider: &str,
) -> Option<Response> {
    if config::serves(caps, model, capability) != Some(false) {
        return None;
    }
    let declared: Vec<&str> = caps
        .iter()
        .filter(|(name, pats)| name.as_str() != capability && pats.iter().any(|p| fnmatch(p, model)))
        .map(|(name, _)| name.as_str())
        .collect();
    Some(err_response(
        StatusCode::BAD_REQUEST,
        format!(
            "model '{provider}/{model}' is declared as {} for this inferencer, not '{capability}' — \
             it cannot serve this endpoint. Adjust [inferencers.{provider}.capabilities] if that \
             is wrong.",
            declared.join("/")
        ),
        "invalid_request_error",
    ))
}

/// Capability tokens that positively mean "this model does not serve chat".
///
/// Deliberately an exclusion set rather than requiring a positive chat marker. A backend's word
/// for "chat" is vendor vocabulary that may not be stable across vendors or firmware — this
/// device says `main` — whereas `embedding` and `rerank` are unambiguous statements of what the
/// model is FOR. Keying on those keeps the fail-open direction and does not depend on the
/// positive vocabulary staying put.
const NON_CHAT_CAPABILITIES: &[&str] = &["embedding", "rerank", "reranking", "tts", "asr", "speech"];

/// What the backend itself said this model can do, mapped onto the capability being asked for.
/// `None` = it said nothing useful, which means unknown and therefore eligible.
fn discovered_serves(discovered: &pool::ModelCapabilities, model: &str, capability: &str) -> Option<bool> {
    let tokens = discovered.get(model)?;
    if tokens.iter().any(|t| t == capability) {
        return Some(true);
    }
    if capability == "chat" && tokens.iter().any(|t| NON_CHAT_CAPABILITIES.contains(&t.as_str())) {
        return Some(false);
    }
    None
}

/// Which residents may satisfy `<provider>/default`, best first.
///
/// A device's residency is device-wide, not per-inferencer: it lists everything loaded, including
/// models this route was never configured to serve. Handing that raw to the resolver let `default`
/// pick an embedder for a chat request — which the backend then rejects, since it is loaded but not
/// servable on that endpoint.
///
/// Two rules, both deliberately vendor-neutral (no capability metadata required — that is issue
/// #20, and it will refine this rather than replace it):
///
/// 1. **If the inferencer declares `models`, only those are candidates.** A model the operator
///    never listed for this route is not a legitimate answer for that route's `default`.
/// 2. **Order deterministically:** a configured `virtual.default` first if it is resident, then
///    lexicographic. Previously the order came from `HashMap` iteration — `reconcile` stamps every
///    newly-discovered resident with the SAME `last_used`, so `snapshot`'s sort is a no-op and the
///    winner was decided by Rust's per-process hash seed. That made `default` **nondeterministic
///    across restarts**: stable within one process, different in the next.
///
/// Fails open: with no `models` configured, every resident stays a candidate, so a backend that
/// never declares a catalog behaves as before.
fn default_candidates(
    inf: &engine::Inferencer,
    caps: &config::CapabilityMap,
    capability: &str,
    residency: pool::Residency,
) -> Vec<String> {
    let pool::Residency { models: residency, capabilities: discovered, current } = residency;
    // Drop residents POSITIVELY known not to serve this endpoint. `default` is never asked in the
    // abstract — it is asked at an endpoint — so an embedding or rerank model is not a tied
    // candidate for a chat request, it is not a candidate at all. Unknown stays eligible, so a
    // backend that publishes nothing and has no declarations behaves exactly as before.
    //
    // Config wins over discovery where both speak: an operator correcting a backend needs the
    // last word, and a backend's self-report has already been observed wrong in this deployment.
    let residency: Vec<String> = residency
        .into_iter()
        .filter(|m| {
            config::serves(caps, m, capability)
                .or_else(|| discovered_serves(&discovered, m, capability))
                != Some(false)
        })
        .collect();
    let mut candidates: Vec<String> = if inf.models.is_empty() {
        residency
    } else {
        let allowed: std::collections::HashSet<&str> = inf.models.iter().map(String::as_str).collect();
        let filtered: Vec<String> = residency.iter().filter(|m| allowed.contains(m.as_str())).cloned().collect();
        // If nothing resident is a configured model, fall back to the unfiltered set rather than
        // reporting "nothing loaded": the caller's own `virtual.default` fallback and the
        // load-on-demand path both handle that better than an empty list does.
        //
        // But SAY SO. In this state `default` can only pick a model this route never declared —
        // possibly one that cannot serve the endpoint at all — and rule 2 makes that choice
        // deterministic, so the route fails the same way every time rather than intermittently.
        // Reliably wrong is easier to diagnose than usually wrong, but only if it is announced;
        // silently it just looks like the router is broken.
        if filtered.is_empty() {
            // ONLY when we actually saw the device. An empty set from a FAILED read means we are
            // blind, not that the operator's `models` list is wrong — and telling someone to fix a
            // config that is fine sends them to the wrong place, especially when they have just
            // edited it. The read failure logs its own specific line; that one is enough.
            if current {
                warn_fail_open(&inf.name);
            }
            // Note this falls open to the CAPABILITY-filtered set, not to every resident: where a
            // backend publishes enough for us to know, an unpredicted-but-capable model is served
            // transparently instead of the request 503ing on an embedder.
            residency
        } else {
            filtered
        }
    };
    candidates.sort();
    if let Some(preferred) = inf.virtual_models.get("default") {
        if let Some(i) = candidates.iter().position(|m| m == preferred) {
            candidates.swap(0, i);
            candidates[1..].sort();
        }
    }
    candidates
}

/// Resolve → load-on-demand → gate → dispatch, for a management-capable inferencer.
/// `Backpressure` => 503 + `Retry-After`; device errors => 502. A direct port of
/// `router.py::_passthrough_pooled`.
async fn passthrough_pooled(
    inf: &engine::Inferencer,
    caps: &config::CapabilityMap,
    manager: &Arc<pool::DeviceModelManager>,
    gate: &pool::Gate,
    body: &Value,
    bare: &str,
) -> Response {
    // `default` is the ONLY resolution that depends on what the device is running, and it is the
    // one decision that is expensive to get wrong: resolving it against our own bookkeeping 400s
    // with a model loaded, or falls through to the `[virtual]` table and evicts a perfectly good
    // model to load its entry (issue #26). So read through to the device here and let it
    // arbitrate — woollama is not the only consumer of its management API, so our view is a cache,
    // never a ledger. A concrete model id or an alias needs no device round trip.
    let loaded = if bare == "default" {
        default_candidates(
            inf,
            caps,
            "chat",
            manager.residency().await,
        )
    } else {
        manager.snapshot()
    };
    let default = inf.virtual_models.get("default").map(String::as_str);
    let real = match engine::resolver::resolve(bare, &inf.virtual_models, &loaded, default) {
        Ok(r) => r,
        Err(e) => return err_response(StatusCode::BAD_REQUEST, e.0, "invalid_request_error"),
    };

    let mut fwd = body.clone();
    fwd["model"] = json!(real);

    let headers = match inf.auth_headers() {
        Ok(h) => h,
        Err(e) => return engine_err_response(e),
    };

    let slot = match gate.enter(&real).await {
        Ok(s) => s,
        Err(pool::PoolError::Backpressure(secs)) => return backpressure_response(secs),
        Err(pool::PoolError::Device(msg)) => {
            // Matches Python's `_error(f"device error: {e}", "server_error", 502)`
            // in `router.py::_passthrough_pooled`.
            return engine_err_response(EngineError::new(format!("device error: {msg}"), "server_error", 502));
        }
    };

    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    if stream {
        let resp = match forward_post(inf.chat_url(), &fwd, &headers, 180).await {
            Ok(r) => r,
            Err(e) => return e,
        };
        if resp.status().as_u16() >= 400 {
            return relay_json(resp).await;
        }
        // Hold `slot` for the lifetime of the stream body: it moves into the
        // generator and drops only once the upstream stream is exhausted (or the
        // body is dropped early, e.g. a client disconnect) — releasing the
        // in-flight counter and the concurrency permit at that point, never before.
        let body_stream = stream! {
            let _slot = slot;
            let mut bs = resp.bytes_stream();
            while let Some(chunk) = bs.next().await {
                yield chunk;
            }
        };
        return sse_response(Body::from_stream(body_stream));
    }

    fwd["stream"] = json!(false);
    let result = match forward_post(inf.chat_url(), &fwd, &headers, 180).await {
        Ok(resp) => relay_json(resp).await,
        Err(e) => e,
    };
    // `slot` drops here (end of scope), after the dispatch has completed.
    result
}

// --- POST /v1/responses (stateless, non-stream) -------------------------------

async fn responses_create(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let input = body.get("input").cloned().unwrap_or_else(|| json!(""));
    let messages = match responses::parse_input(&input) {
        Ok(m) => m,
        Err(e) => return err_response(StatusCode::BAD_REQUEST, e, "invalid_request_error"),
    };

    let nonnull = |k: &str| body.get(k).is_some_and(|v| !v.is_null());
    let stateful = body.get("store").and_then(Value::as_bool).unwrap_or(false)
        || nonnull("conversation")
        || nonnull("previous_response_id")
        || nonnull("key");
    if stateful {
        if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
            return err_response(
                StatusCode::BAD_REQUEST,
                "streaming is not supported for STATEFUL /v1/responses conversations",
                "invalid_request_error",
            );
        }
        return responses_stateful(&state, &body, &model, &json!(messages)).await;
    }
    if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        let source: BoxStream<'static, Result<String, EngineError>> =
            if let Some(name) = model.strip_prefix("woollama/") {
                let Some(recipe) = state.recipes.get(name) else {
                    return err_response(StatusCode::NOT_FOUND, format!("unknown recipe '{name}'"), "not_found");
                };
                if let Some(cc_model) = recipe.inferencer.strip_prefix("claude-code/") {
                    // claude-code is non-streaming: the answer is one delta.
                    let text = match run_claude_recipe(&state, recipe, &json!(messages), cc_model).await {
                        Ok(resp) => resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string(),
                        Err(e) => return engine_err_response(e),
                    };
                    futures::stream::once(async move { Ok::<String, EngineError>(text) }).boxed()
                } else {
                    let recipe_val = recipe.to_value();
                    let provider: Arc<dyn engine::ToolProvider> =
                        Arc::new(mcp_registry::RegistryToolProvider { reg: state.registry.clone() });
                    let setup = match engine::build_setup(&recipe_val, &json!(messages), provider, None, None, Some(&state.inferencers)) {
                        Ok(s) => s,
                        Err(e) => return engine_err_response(e),
                    };
                    engine::events_stream(setup, true)
                        .filter_map(|item| async move {
                            match item {
                                Ok(engine::Event::Delta(c)) => Some(Ok(c)),
                                Ok(_) => None,
                                Err(e) => Some(Err(e)),
                            }
                        })
                        .boxed()
                }
            } else {
                let options = body.get("options").cloned();
                let req = match engine::build_request(&model, json!(messages), options, None, None, None, true) {
                    Ok(r) => r,
                    Err(e) => return engine_err_response(e),
                };
                engine::complete_stream_events(req).boxed()
            };
        return responses_stream(source, model).await;
    }

    let resp_id = responses::new_id("resp");

    // woollama/<recipe> → orchestrate; extract the final assistant text.
    if let Some(name) = model.strip_prefix("woollama/") {
        let Some(recipe) = state.recipes.get(name) else {
            return err_response(StatusCode::NOT_FOUND, format!("unknown recipe '{name}'"), "not_found");
        };
        return match orchestrate_recipe(&state, recipe, &json!(messages)).await {
            Ok(resp) => {
                let text = resp["choices"][0]["message"]["content"].as_str().unwrap_or("");
                Json(responses::build_response(&resp_id, &model, text)).into_response()
            }
            Err(e) => engine_err_response(e),
        };
    }

    // Stateless inferencer turn — the engine's complete handles native num_ctx.
    let options = body.get("options").cloned();
    let req = match engine::build_request(&model, json!(messages), options, None, None, None, false) {
        Ok(r) => r,
        Err(e) => return engine_err_response(e),
    };
    match engine::run_complete(req).await {
        Ok(text) => Json(responses::build_response(&resp_id, &model, &text)).into_response(),
        Err(e) => engine_err_response(e),
    }
}

// --- stateful conversations (slice 6a) ----------------------------------------

fn no_stateful_backend_msg(model: &str) -> String {
    format!(
        "no stateful backend for model '{model}': only claude-code (claude-resume) has one \
         in this build (managed-agents + store-backed are later slices). Use store:false \
         (the caller owns history)."
    )
}

/// Run one stateful /v1/responses turn: resolve/create the conversation handle, run the
/// turn on its backend under a per-conversation write lock, return the Responses object
/// carrying the conversation id. Slice 6a: the claude-resume backend only.
async fn responses_stateful(state: &AppState, body: &Value, model: &str, messages: &Value) -> Response {
    let conv_id_param = body.get("conversation").and_then(Value::as_str);
    let prev = body.get("previous_response_id").and_then(Value::as_str);
    let key = body.get("key").and_then(Value::as_str);

    // Resolve or create the conversation handle (explicit id wins, then prev, then key,
    // else a new one whose backend is derived from the model).
    let conv = {
        let mut t = state.conversations.table.lock().await;
        if let Some(cid) = conv_id_param {
            match t.get(cid) {
                Some(c) => c,
                None => return err_response(StatusCode::NOT_FOUND, format!("unknown conversation '{cid}'"), "not_found"),
            }
        } else if let Some(p) = prev {
            match t.by_response(p) {
                Some(c) => c,
                None => return err_response(StatusCode::NOT_FOUND, format!("unknown previous_response_id '{p}'"), "not_found"),
            }
        } else {
            let Some(backend) = state.backend_for_model(model) else {
                return err_response(StatusCode::NOT_IMPLEMENTED, no_stateful_backend_msg(model), "not_implemented");
            };
            match key {
                Some(k) => t.get_or_create_by_alias(k, backend, model),
                None => t.create(backend, model, json!({}), None, None),
            }
        }
    };

    // One writer per conversation: hold the per-conv lock across the turn (but NOT the
    // table lock, which only guards brief reads/writes — each backend turn does its own).
    let lock = state.conversations.conv_lock(&conv.id).await;
    let _guard = lock.lock().await;

    // Re-read the row now that we hold the per-conv lock: the snapshot above was taken under
    // the table lock, which we then released, so a concurrent same-conversation turn could
    // have advanced status/pending/native_id. Acting on the stale snapshot would re-resolve an
    // already-answered managed-agents custom_tool_use_id (double-resume).
    let conv = {
        let t = state.conversations.table.lock().await;
        t.get(&conv.id).unwrap_or(conv)
    };

    let options = body.get("options").cloned();
    let turn: Result<(String, Option<Value>), EngineError> = match conv.backend.as_str() {
        "claude-resume" => claude_resume_turn(state, &conv.id, &conv.model, messages).await.map(|t| (t, None)),
        "store-backed" => store_backed_turn(state, &conv.id, &conv.model, messages, options).await.map(|t| (t, None)),
        "managed-agents" => managed_agents_turn(state, &conv, messages).await,
        other => {
            return err_response(
                StatusCode::NOT_IMPLEMENTED,
                format!("the '{other}' backend is not in the Rust server"),
                "not_implemented",
            )
        }
    };
    let (text, required_action) = match turn {
        Ok(x) => x,
        Err(e) => return engine_err_response(e),
    };

    let resp_id = responses::new_id("resp");
    {
        let mut t = state.conversations.table.lock().await;
        t.record_response(&conv.id, &resp_id);
    }
    let status = if required_action.is_some() { "requires_action" } else { "completed" };
    Json(responses::build_response_stateful(&resp_id, &conv.model, &text, &conv.id, status, required_action))
        .into_response()
}

/// The latest user message text — for a stateful turn the backend already owns prior
/// history, so woollama sends only the new user input.
fn latest_user_text(messages: &Value) -> String {
    let Some(arr) = messages.as_array() else { return String::new() };
    for m in arr.iter().rev() {
        if m.get("role").and_then(Value::as_str) == Some("user") {
            return m.get("content").and_then(Value::as_str).unwrap_or("").to_string();
        }
    }
    arr.last().and_then(|m| m.get("content").and_then(Value::as_str)).unwrap_or("").to_string()
}

/// One managed-agents turn: resume a paused session with the answer (if awaiting input),
/// else run a fresh turn (creating the hosted session lazily). Returns the text + the
/// `required_action` payload when the agent paused on ask_user.
async fn managed_agents_turn(state: &AppState, conv: &conversations::Conversation, messages: &Value) -> Result<(String, Option<Value>), EngineError> {
    let ma = &state.managed_agents;
    let to_err = |e: managed_agents::ManagedAgentsError| EngineError::new(format!("managed-agents backend: {e}"), "server_error", 502);

    let mut native_id = conv.native_id.clone();
    let turn = if conv.status == "awaiting_input" && conv.pending_tool_use_id.is_some() {
        ma.answer_turn(
            native_id.as_deref().unwrap_or(""),
            conv.pending_tool_use_id.as_deref().unwrap(),
            &latest_user_text(messages),
        )
        .await
        .map_err(to_err)?
    } else {
        if native_id.is_none() {
            native_id = Some(ma.create_session(&conv.model, conv.title.as_deref(), &conv.metadata).await.map_err(to_err)?);
        }
        ma.run_turn(native_id.as_deref().unwrap(), &latest_user_text(messages)).await.map_err(to_err)?
    };

    let (status, required_action, pending_id) = match &turn.pending {
        Some(p) => (
            "awaiting_input".to_string(),
            Some(json!({"type": "ask_user", "question": p.input})),
            Some(p.id.clone()),
        ),
        None => ("idle".to_string(), None, None),
    };
    state.conversations.table.lock().await.set_managed(&conv.id, native_id, status, required_action.clone(), pending_id);
    Ok((turn.text, required_action))
}

/// One claude-resume turn: ensure a stable workdir, `--resume` the session, persist the
/// captured/echoed session_id.
async fn claude_resume_turn(state: &AppState, conv_id: &str, model: &str, messages: &Value) -> Result<String, EngineError> {
    let (mut native_id, mut workdir) = {
        let t = state.conversations.table.lock().await;
        let c = t.get(conv_id);
        (c.as_ref().and_then(|c| c.native_id.clone()), c.and_then(|c| c.workdir.clone()))
    };
    if workdir.is_none() {
        let dir = std::env::temp_dir().join(format!("woollama-conv-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).map_err(|e| EngineError::new(e.to_string(), "server_error", 500))?;
        workdir = Some(dir.to_string_lossy().to_string());
        // Record the workdir BEFORE the turn so a first-turn failure still leaves it referenced
        // for teardown to reclaim (else an errored first turn leaks an orphaned
        // /tmp/woollama-conv-* dir with no table reference). A retry reuses the same dir.
        state.conversations.table.lock().await.set_native(conv_id, native_id.clone(), workdir.clone());
    }
    let cc_model = model.strip_prefix("claude-code/").unwrap_or("");
    let (resp, sid) = claude_code::run_resumable(messages, cc_model, native_id.as_deref(), workdir.as_deref().unwrap())
        .await
        .map_err(|e| EngineError::new(format!("claude-resume backend: {e}"), "server_error", 502))?;
    if sid.is_some() {
        native_id = sid;
    }
    let text = resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
    state.conversations.table.lock().await.set_native(conv_id, native_id, workdir);
    Ok(text)
}

/// One store-backed turn: the external store owns the transcript; woollama assembles
/// prior + new, runs STATELESS inference, and writes the turn back.
async fn store_backed_turn(state: &AppState, conv_id: &str, model: &str, messages: &Value, options: Option<Value>) -> Result<String, EngineError> {
    let store = state.store.clone().ok_or_else(|| EngineError::new("no conversation store configured", "server_error", 500))?;
    let mut native_id = {
        let t = state.conversations.table.lock().await;
        t.get(conv_id).and_then(|c| c.native_id.clone())
    };
    if native_id.is_none() {
        native_id = Some(store.create().await?); // the store mints the thread
    }
    let tid = native_id.clone().unwrap();
    let mut combined = store.get(&tid).await?; // bytes owned by the store
    combined.extend(messages.as_array().cloned().unwrap_or_default());
    let answer = complete_stateless(state, model, &json!(combined), options).await?;
    let mut to_append = messages.as_array().cloned().unwrap_or_default();
    to_append.push(json!({"role": "assistant", "content": answer}));
    store.append(&tid, &json!(to_append)).await?; // write the turn back
    state.conversations.table.lock().await.set_native(conv_id, native_id, None);
    Ok(answer)
}

/// Run one stateless turn and return the assistant text — routes by model exactly like
/// /v1/chat/completions (woollama/<recipe> → orchestrate; a known inferencer → complete,
/// honoring native num_ctx via options). The inference fn for store-backed turns.
async fn complete_stateless(state: &AppState, model: &str, messages: &Value, options: Option<Value>) -> Result<String, EngineError> {
    if let Some(name) = model.strip_prefix("woollama/") {
        let recipe = state
            .recipes
            .get(name)
            .ok_or_else(|| EngineError::new(format!("unknown recipe '{name}'"), "not_found", 404))?;
        let resp = orchestrate_recipe(state, recipe, messages).await?;
        Ok(resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
    } else {
        let req = engine::build_request(model, messages.clone(), options, None, None, None, false)?;
        engine::run_complete(req).await
    }
}

async fn conversations_create(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let model = body.get("model").and_then(Value::as_str).unwrap_or("");
    if model.is_empty() {
        return err_response(StatusCode::BAD_REQUEST, "`model` is required to create a conversation", "invalid_request_error");
    }
    let backend = body
        .get("backend")
        .and_then(Value::as_str)
        .map(String::from)
        .or_else(|| state.backend_for_model(model).map(String::from));
    let Some(backend) = backend.filter(|b| b == "claude-resume" || b == "store-backed" || b == "managed-agents") else {
        return err_response(StatusCode::NOT_IMPLEMENTED, no_stateful_backend_msg(model), "not_implemented");
    };
    let key = body.get("key").and_then(Value::as_str).map(String::from);
    let metadata = body.get("metadata").cloned().unwrap_or_else(|| json!({}));
    let title = body.get("title").and_then(Value::as_str).map(String::from);

    let mut t = state.conversations.table.lock().await;
    if let Some(k) = &key {
        if let Some(existing) = t.by_alias(k) {
            return (StatusCode::OK, Json(existing.to_object())).into_response();
        }
    }
    let conv = t.create(&backend, model, metadata, title, key);
    (StatusCode::CREATED, Json(conv.to_object())).into_response()
}

async fn conversations_list(State(state): State<Arc<AppState>>) -> Json<Value> {
    let t = state.conversations.table.lock().await;
    let data: Vec<Value> = t.list().iter().map(|c| c.to_object()).collect();
    Json(json!({"object": "list", "data": data}))
}

async fn conversations_get(State(state): State<Arc<AppState>>, Path(conv_id): Path<String>) -> Response {
    let t = state.conversations.table.lock().await;
    match t.get(&conv_id) {
        Some(c) => Json(c.to_object()).into_response(),
        None => err_response(StatusCode::NOT_FOUND, format!("unknown conversation '{conv_id}'"), "not_found"),
    }
}

async fn conversations_items(State(state): State<Arc<AppState>>, Path(conv_id): Path<String>) -> Response {
    let conv = {
        let t = state.conversations.table.lock().await;
        match t.get(&conv_id) {
            Some(c) => c,
            None => return err_response(StatusCode::NOT_FOUND, format!("unknown conversation '{conv_id}'"), "not_found"),
        }
    };
    // store-backed + managed-agents serve the transcript (from the store / Anthropic's
    // event log); claude-resume has no `history` (a later driver slice) → 501.
    if conv.backend == "store-backed" || conv.backend == "managed-agents" {
        let msgs: Vec<Value> = match conv.backend.as_str() {
            "store-backed" => {
                let Some(store) = state.store.clone() else {
                    return engine_err_response(EngineError::new("no conversation store configured", "server_error", 500));
                };
                match &conv.native_id {
                    Some(tid) => match store.get(tid).await {
                        Ok(m) => m,
                        Err(e) => return engine_err_response(e),
                    },
                    None => Vec::new(),
                }
            }
            _ => match &conv.native_id {
                Some(sid) => match state.managed_agents.history(sid).await {
                    Ok(m) => m,
                    Err(e) => return engine_err_response(EngineError::new(format!("managed-agents backend: {e}"), "server_error", 502)),
                },
                None => Vec::new(),
            },
        };
        let data: Vec<Value> = msgs.iter().map(responses::item_object).collect();
        let first_id = data.first().and_then(|x| x.get("id").cloned()).unwrap_or(Value::Null);
        let last_id = data.last().and_then(|x| x.get("id").cloned()).unwrap_or(Value::Null);
        return Json(json!({
            "object": "list", "data": data,
            "first_id": first_id, "last_id": last_id, "has_more": false,
        }))
        .into_response();
    }
    err_response(
        StatusCode::NOT_IMPLEMENTED,
        format!("conversation transcript items are not available for the '{}' backend yet", conv.backend),
        "not_implemented",
    )
}

async fn conversations_delete(State(state): State<Arc<AppState>>, Path(conv_id): Path<String>) -> Response {
    let conv = {
        let t = state.conversations.table.lock().await;
        t.get(&conv_id)
    };
    let Some(conv) = conv else {
        return err_response(StatusCode::NOT_FOUND, format!("unknown conversation '{conv_id}'"), "not_found");
    };
    // Backend teardown (best-effort): claude-resume removes the per-conversation workdir
    // (the on-disk Claude session is the user's data, left intact); store-backed tells the
    // external store to drop the thread.
    match conv.backend.as_str() {
        "claude-resume" => {
            if let Some(wd) = &conv.workdir {
                let _ = std::fs::remove_dir_all(wd);
            }
        }
        "store-backed" => {
            if let (Some(store), Some(tid)) = (state.store.clone(), conv.native_id.clone()) {
                let _ = store.delete(&tid).await;
            }
        }
        "managed-agents" => {
            if let Some(sid) = &conv.native_id {
                let _ = state.managed_agents.delete_session(sid).await;
            }
        }
        _ => {}
    }
    {
        let mut t = state.conversations.table.lock().await;
        t.remove(&conv_id);
    }
    Json(json!({"id": conv_id, "object": "conversation.deleted", "deleted": true})).into_response()
}

#[cfg(test)]
mod default_candidate_tests {
    use super::*;

    fn inf(models: &[&str], virtual_default: Option<&str>) -> engine::Inferencer {
        let mut virtual_models = std::collections::BTreeMap::new();
        if let Some(d) = virtual_default {
            virtual_models.insert("default".to_string(), d.to_string());
        }
        engine::Inferencer {
            name: "dev".into(),
            base_url: "http://x/v1".into(),
            api_key_env: None,
            extra_body: serde_json::json!({}),
            models: models.iter().map(|s| s.to_string()).collect(),
            discover: false,
            model_patterns: Vec::new(),
            management_url: Some("http://x".into()),
            management_protocol: None,
            parallel: 1,
            pool_max: None,
            queue_max: None,
            queue_timeout: 30.0,
            virtual_models,
        }
    }

    fn no_caps() -> config::CapabilityMap {
        config::CapabilityMap::new()
    }

    fn current(models: Vec<String>) -> pool::Residency {
        pool::Residency { models, capabilities: pool::ModelCapabilities::new(), current: true }
    }

    /// Residency as the DEVICE describes it — the shape that removes the treadmill entirely,
    /// because the exclusion set is discovered rather than configured.
    fn discovered(models: Vec<String>, caps: &[(&str, &[&str])]) -> pool::Residency {
        let mut capabilities = pool::ModelCapabilities::new();
        for (id, tokens) in caps {
            capabilities.insert(id.to_string(), tokens.iter().map(|t| t.to_string()).collect());
        }
        pool::Residency { models, capabilities, current: true }
    }

    fn residency_models() -> Vec<String> {
        // Device-wide residency, as a real device reports it: a chat model plus two that cannot
        // serve chat at all.
        ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3.6-35B-A3B-turbo"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn a_model_this_route_never_declared_is_not_a_candidate() {
        // The reported failure: `default` picked the embedder, which was never in this
        // inferencer's `models` list, and the backend rejected it as "not loaded" — it IS loaded,
        // just not servable on the chat endpoint.
        let got = default_candidates(&inf(&["Qwen/Qwen3.6-35B-A3B-turbo"], None), &no_caps(), "chat", current(residency_models()));
        assert_eq!(got, vec!["Qwen/Qwen3.6-35B-A3B-turbo".to_string()]);
    }

    #[test]
    fn selection_is_deterministic_not_hash_ordered() {
        // `reconcile` stamps every newly-discovered resident with the SAME last_used, so
        // `snapshot`'s sort was a no-op and the winner came from HashMap iteration order — i.e.
        // Rust's per-process hash seed. `default` was therefore stable within a process and
        // different in the next one.
        let i = inf(&[], None);
        let first = default_candidates(&i, &no_caps(), "chat", current(residency_models()));
        let mut shuffled = residency_models();
        shuffled.reverse();
        assert_eq!(first, default_candidates(&i, &no_caps(), "chat", current(shuffled)), "same residency ⇒ same choice, any input order");
    }

    #[test]
    fn a_resident_virtual_default_wins_the_tiebreak() {
        let got = default_candidates(
            &inf(&[], Some("Qwen/Qwen3.6-35B-A3B-turbo")),
            &no_caps(),
            "chat",
            current(residency_models()),
        );
        assert_eq!(got[0], "Qwen/Qwen3.6-35B-A3B-turbo", "a configured default that IS resident wins");
    }

    #[test]
    fn a_declared_mismatch_is_refused_before_dispatch() {
        // #20's motivating bug is a backend that answers an unsupported request by taking the
        // whole model service down. Where the operator has declared the model's purpose, refuse
        // it here and return a 400 naming what it IS, rather than letting the backend decide.
        let mut caps = config::CapabilityMap::new();
        caps.insert("embedding".into(), vec!["*Embedding*".into()]);
        assert!(
            reject_wrong_capability(&caps, "Qwen/Qwen3-Embedding-0.6B", "chat", "dev").is_some(),
            "an embedder must not reach the chat endpoint"
        );
        assert!(
            reject_wrong_capability(&caps, "Qwen/Qwen3-Embedding-0.6B", "embedding", "dev").is_none(),
            "and must still serve the endpoint it IS for"
        );
    }

    #[test]
    fn an_undeclared_model_is_never_refused() {
        // Fail open, everywhere. A model nobody has described must dispatch exactly as before —
        // this check may only ever turn a KNOWN mismatch into a clear error, never invent one.
        let mut caps = config::CapabilityMap::new();
        caps.insert("embedding".into(), vec!["*Embedding*".into()]);
        assert!(reject_wrong_capability(&caps, "zai-org/GLM-4.7-Flash", "chat", "dev").is_none());
        assert!(reject_wrong_capability(&config::CapabilityMap::new(), "anything", "chat", "dev").is_none());
    }

    #[test]
    fn a_backends_own_capability_report_excludes_non_chat_residents() {
        // FIXTURE CONSTRAINT: the embedder must sort BEFORE the chat model, or the filter is not
        // what makes this pass. `Qwen/Qwen3-Embedding` < `Qwen/Qwen3.6-...` because '-' (0x2D)
        // precedes '.' (0x2E) — so without the capability filter the embedder wins and the test
        // fails, which is the point.
        //
        // Zero configuration. The device publishes `capabilities` per resident in the same payload
        // woollama already fetches, so the exclusion set is DISCOVERED — which is what removes the
        // treadmill: nobody has to enumerate models a peer might load.
        let got = default_candidates(
            &inf(&[], None),
            &no_caps(),
            "chat",
            discovered(
                residency_models(),
                &[
                    ("Qwen/Qwen3-Embedding-0.6B", &["embedding"]),
                    ("Qwen/Qwen3-Reranker-0.6B", &["rerank"]),
                    ("Qwen/Qwen3.6-35B-A3B-turbo", &["main"]),
                ],
            ),
        );
        assert_eq!(got, vec!["Qwen/Qwen3.6-35B-A3B-turbo".to_string()]);
    }

    #[test]
    fn an_unpredicted_model_the_backend_calls_chat_is_served() {
        // The production failure, with discovery: a peer loads a model nobody listed. It is not
        // excluded (no non-chat token), so it is served transparently instead of the request
        // 503ing on an embedder.
        //
        // FIXTURE CONSTRAINT — do not "simplify" the ids. The unknown chat model's id must sort
        // AFTER the non-chat resident, because the fallback ordering is lexicographic: with
        // `Qwen/Qwen3-Coder-...` instead of `zai-org/...` this test passes even WITHOUT the
        // capability filter, since C sorts before E and the chat model wins by luck of the
        // alphabet. That near-miss happened on real hardware.
        let models = vec!["Qwen/Qwen3-Embedding-0.6B".to_string(), "zai-org/GLM-4.7-Flash".to_string()];
        let got = default_candidates(
            &inf(&[], None),
            &no_caps(),
            "chat",
            discovered(
                models,
                &[("Qwen/Qwen3-Embedding-0.6B", &["embedding"]), ("zai-org/GLM-4.7-Flash", &["main"])],
            ),
        );
        assert_eq!(got, vec!["zai-org/GLM-4.7-Flash".to_string()]);
    }

    #[test]
    fn config_overrides_what_the_backend_says_about_itself() {
        // An operator correcting a backend needs the last word — this deployment has already
        // caught a device's self-report wrong by a factor of 26 on eviction cost.
        let mut caps = config::CapabilityMap::new();
        caps.insert("chat".into(), vec!["*Embedding*".into()]);
        let got = default_candidates(
            &inf(&[], None),
            &caps,
            "chat",
            discovered(
                vec!["Qwen/Qwen3-Embedding-0.6B".to_string()],
                &[("Qwen/Qwen3-Embedding-0.6B", &["embedding"])],
            ),
        );
        assert_eq!(got, vec!["Qwen/Qwen3-Embedding-0.6B".to_string()], "config wins over discovery");
    }

    #[test]
    fn fail_open_is_deterministic_and_reported() {
        // A realistic misconfiguration, arrived at honestly: the route declares models, none of
        // which is resident. Falling open is correct — an empty candidate list would be worse —
        // but the choice is then lexicographic among models this route never declared, which on a
        // mixed-residency device means a hard failure EVERY time rather than intermittently.
        // Determinism makes it diagnosable; the warning makes it discoverable.
        let i = inf(&["Qwen/Qwen3-30B-A3B-Instruct"], Some("Qwen/Qwen3-30B-A3B-Instruct"));
        let got = default_candidates(&i, &no_caps(), "chat", current(residency_models()));
        assert_eq!(got.len(), 3, "falls open to every resident rather than reporting none loaded");
        assert_eq!(got, {
            let mut r = residency_models();
            r.sort();
            r
        }, "and does so deterministically — same input, same order, every process");
    }

    #[test]
    fn no_configured_models_keeps_every_resident_a_candidate() {
        // Fail open: a backend that declares no catalog must behave as before.
        assert_eq!(default_candidates(&inf(&[], None), &no_caps(), "chat", current(residency_models())).len(), 3);
        // And if nothing resident is a configured model, don't report an empty set — let the
        // caller's virtual.default / load-on-demand path handle it.
        assert_eq!(default_candidates(&inf(&["Not/Resident"], None), &no_caps(), "chat", current(residency_models())).len(), 3);
    }
}

#[cfg(test)]
mod delegate_config_tests {
    use super::*;
    use crate::config::{McpServerSpec, StdioSpec};

    fn specs_with(name: &str, spec: McpServerSpec) -> HashMap<String, McpServerSpec> {
        let mut specs = HashMap::new();
        specs.insert(name.to_string(), spec);
        specs
    }

    #[test]
    fn delegate_config_forwards_the_env_block() {
        // A stdio server needing `env` must behave the same in-loop and under claude-code
        // delegation. Dropping it here would make the tool work through woollama's own loop
        // and silently misbehave when Claude runs it — the worst kind of divergence, because
        // both paths report success.
        let mut env = HashMap::new();
        env.insert("GIT_AUTHOR_NAME".to_string(), "woollama".to_string());
        let specs =
            specs_with("git", McpServerSpec::Stdio(StdioSpec { command: "git-mcp".into(), args: vec![], env }));
        let out = referenced_mcp_servers(&specs, &["git.log".to_string()]).unwrap();
        assert_eq!(out["git"]["env"]["GIT_AUTHOR_NAME"], "woollama");
        assert_eq!(out["git"]["command"], "git-mcp");
    }

    #[test]
    fn an_unconfigured_server_is_still_an_error() {
        let specs = specs_with("git", McpServerSpec::Stdio(StdioSpec { command: "g".into(), args: vec![], env: HashMap::new() }));
        let err = referenced_mcp_servers(&specs, &["other.tool".to_string()]).unwrap_err();
        assert!(err.message.contains("other"), "must name the missing server: {}", err.message);
    }
}

#[cfg(test)]
mod fnmatch_tests {
    use super::fnmatch;

    #[test]
    fn glob_star_and_question() {
        assert!(fnmatch("gpt-4*", "gpt-4o"));
        assert!(fnmatch("keep-*", "keep-this"));
        assert!(!fnmatch("keep-*", "drop-that"));
        assert!(fnmatch("*", "anything"));
        assert!(fnmatch("q?en", "qwen"));
        assert!(!fnmatch("q?en", "qween"));
        assert!(fnmatch("exact", "exact"));
        assert!(!fnmatch("exact", "exacto"));
        assert!(fnmatch("a*b*c", "axxbyyc"));
    }
}

#[cfg(test)]
mod address_tests {
    use super::parse_tcp_address;

    #[test]
    fn parses_ipv4_ipv6_host_and_port_forms() {
        // IPv4 / host:port
        assert_eq!(parse_tcp_address("127.0.0.1:8080"), ("127.0.0.1".into(), 8080));
        assert_eq!(parse_tcp_address("0.0.0.0:9000"), ("0.0.0.0".into(), 9000));
        assert_eq!(parse_tcp_address("localhost:1234"), ("localhost".into(), 1234));
        // bare IP → ephemeral port
        assert_eq!(parse_tcp_address("127.0.0.1"), ("127.0.0.1".into(), 0));
        // :port → loopback
        assert_eq!(parse_tcp_address(":7000"), ("127.0.0.1".into(), 7000));
        // bare host
        assert_eq!(parse_tcp_address("myhost"), ("myhost".into(), 0));

        // IPv6 — the forms the old first-`:`-split broke:
        assert_eq!(parse_tcp_address("[::1]:8080"), ("::1".into(), 8080));
        assert_eq!(parse_tcp_address("[::]:8080"), ("::".into(), 8080));
        assert_eq!(parse_tcp_address("::1"), ("::1".into(), 0)); // no longer downgraded to IPv4
        assert_eq!(parse_tcp_address("[::1]"), ("::1".into(), 0));
        // the exact old-panic input never yields the bogus host `"["`
        assert_ne!(parse_tcp_address("[::]:8080").0, "[");
    }
}
