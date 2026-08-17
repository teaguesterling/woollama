//! The downstream MCP registry — one rmcp **client** per configured server (spawned as
//! a child process over stdio), and the `RegistryToolProvider` that adapts it to the
//! engine's `ToolProvider` seam so the recipe loop can dispatch MCP tools.
//!
//! Mirrors Python `manager.Registry` / `RegistryToolProvider`. (The asyncio
//! queue-marshaling workaround the Python version needs doesn't apply here: rmcp's
//! `Peer` is a cheap, Send+Sync, clonable handle.)

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::model::{CallToolRequestParams, CallToolResult, Tool};
use rmcp::service::RoleClient;
use rmcp::transport::streamable_http_client::{StreamableHttpClientTransport, StreamableHttpClientTransportConfig};
use rmcp::transport::TokioChildProcess;
use rmcp::{Peer, ServiceExt};
use serde_json::{json, Value};

use woollama_engine::{EngineError, ToolProvider};

use crate::config::{HttpSpec, McpServerSpec};

struct ServerConn {
    peer: Peer<RoleClient>,
    tools: Vec<Tool>,
}

/// One consistent view of every connected downstream server. Immutable once published: the
/// registry swaps a whole `Arc<Snapshot>` rather than mutating in place, so `servers` and
/// `wire_index` can never be observed disagreeing with each other (a reader that saw a new
/// `wire_index` against an old `servers` would resolve a name to a peer that no longer holds it).
#[derive(Default)]
struct Snapshot {
    servers: HashMap<String, ServerConn>,
    /// Every configured server's health, including those with no entry in `servers`. Carried in
    /// the snapshot so a caller sees health and tools from the same instant.
    health: HashMap<String, (ServerHealth, &'static str)>,
    /// Reverse map: advertised wire name (`mcp__server__tool`) -> (server, bare tool). Built
    /// with the snapshot so dispatch resolves the model's tool_call name unambiguously (no
    /// dot-splitting, and it works for the hashed >64-char fallback too).
    wire_index: HashMap<String, (String, String)>,
}

impl Snapshot {
    /// Build the reverse index for a set of connected servers — the one place a wire name is
    /// derived, so publishing can never disagree with resolving.
    fn new(
        servers: HashMap<String, ServerConn>,
        health: HashMap<String, (ServerHealth, &'static str)>,
    ) -> Snapshot {
        let mut wire_index = HashMap::new();
        for (server, conn) in &servers {
            for t in &conn.tools {
                wire_index.insert(wire_name(server, &t.name), (server.clone(), t.name.to_string()));
            }
        }
        Snapshot { servers, health, wire_index }
    }
}

/// What the router knows about one configured downstream right now.
///
/// Distinct states rather than present/absent, because "absent" conflates two things an operator
/// must act on differently: a peer that is down but will be retried, and one that can never work
/// until a human edits a file. A router that reported a reconnecting server as simply missing
/// would look healthy while its tools were quietly gone.
#[derive(Clone, Debug, PartialEq)]
pub enum ServerHealth {
    /// Connected; its tools are in the current snapshot.
    Connected,
    /// Unreachable, but retried on a backoff. `last_error` is why the most recent attempt failed.
    Retrying { attempts: u32, last_error: String },
    /// Terminal. The transport itself could not be built (an unusable header, an HTTP client that
    /// won't initialize) — a config fault, not a world fault, so retrying would only produce noise
    /// until someone edits the file.
    Failed { reason: String },
}

impl ServerHealth {
    pub fn as_str(&self) -> &'static str {
        match self {
            ServerHealth::Connected => "connected",
            ServerHealth::Retrying { .. } => "retrying",
            ServerHealth::Failed { .. } => "failed",
        }
    }
}

/// One configured downstream, as reported by [`McpRegistry::status`].
#[derive(Clone, Debug)]
pub struct ServerStatus {
    pub name: String,
    pub transport: &'static str,
    pub health: ServerHealth,
    /// Tools currently contributed. Always 0 unless `Connected`.
    pub tools: usize,
}

/// All configured downstream MCP servers, connected and tool-listed.
///
/// Readers clone the `Arc` under a read guard and drop the guard immediately, so no lock is ever
/// held across an `await` and a slow downstream call cannot block a snapshot swap.
pub struct McpRegistry {
    inner: std::sync::RwLock<Arc<Snapshot>>,
}

/// Allow-listed env for a spawned MCP server. Shares `claude_code::CHILD_ENV_ALLOW` (single
/// source of truth) so the downstream-server scrub can't drift from the claude-code one —
/// provider secrets in the daemon env (ANTHROPIC_API_KEY etc.) never reach a tool server.
fn scrubbed_env() -> HashMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| crate::claude_code::CHILD_ENV_ALLOW.contains(&k.as_str()) || k.starts_with("LC_"))
        .collect()
}

/// The scrubbed base env with the spec's `env` block merged OVER it — explicit entries win.
/// The scrub stays a floor: nothing reaches a tool server by inheritance, but an operator can
/// deliberately name a var (including a provider key) in mcp.json. Mirrors the Python
/// reference, where `StdioServerParameters.env` merges over the SDK's safe default env.
fn merged_env(spec_env: &HashMap<String, String>) -> HashMap<String, String> {
    let mut env = scrubbed_env();
    env.extend(spec_env.iter().map(|(k, v)| (k.clone(), v.clone())));
    env
}

/// Build the Streamable-HTTP client transport for a `url`-form downstream server.
///
/// `reqwest::Client` is rmcp's own HTTP client impl (Cargo.toml pins reqwest 0.13 to match), and
/// `axum::http` re-exports the same `http` types rmcp expects — no new dependency for either.
fn http_transport(spec: &HttpSpec) -> Result<StreamableHttpClientTransport<reqwest::Client>, String> {
    use axum::http::{HeaderName, HeaderValue};

    let mut headers = HashMap::new();
    for (k, v) in &spec.headers {
        let name = HeaderName::try_from(k.as_str()).map_err(|e| format!("invalid header name '{k}': {e}"))?;
        // `InvalidHeaderValue`'s Display does not include the offending value, which is what we
        // want — these carry bearer tokens. Do not add the value to this message.
        let value = HeaderValue::from_str(v).map_err(|e| format!("invalid header value for '{k}': {e}"))?;
        headers.insert(name, value);
    }
    let config = StreamableHttpClientTransportConfig::with_uri(spec.url.clone()).custom_headers(headers);
    // `Client::new()` PANICS if a TLS backend or the system resolver config can't be initialized
    // (a minimal container with no CA bundle or a broken /etc/resolv.conf). connect_one runs as a
    // plain future inside build_state, not a spawned task, so that panic would unwind into main
    // and the daemon would never start — defeating the logged-and-skipped contract every other
    // downstream failure honours. Build fallibly and let it be one skipped server.
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| format!("could not build the HTTP client for this downstream: {e}"))?;
    Ok(StreamableHttpClientTransport::with_client(client, config))
}

/// How many levels of `mcp__` namespacing a re-exported tool may carry.
/// `WOOLLAMA_MCP_MAX_NESTING`, default 2. `0` disables the cap.
///
/// This exists because reconnect made unbounded growth reachable. Tool names nest one level per
/// federation hop, and in a mutual topology (A consumes B, B consumes A) each refresh ingests a
/// roster that already contains the previous round's nesting:
///
/// ```text
/// t1  A pulls B  ->  A: [chat, mcp__B__chat]
/// t2  B pulls A  ->  B: [chat, mcp__A__chat, mcp__A__mcp__B__chat]
/// t3  A pulls B  ->  A: [..., mcp__B__mcp__A__mcp__B__chat]
/// ```
///
/// Before reconnect that cost one level per RESTART. On a refresh timer it is one level per tick,
/// forever — a roster that grows without bound, with every name past 64 chars falling back to the
/// hashed form (and so onto issue #22's unstable-hash path). The cap bounds it locally, with no
/// protocol change and nothing to coordinate between routers.
fn max_nesting() -> usize {
    std::env::var("WOOLLAMA_MCP_MAX_NESTING").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
}

/// How many `mcp__` namespace levels a tool name already carries.
fn nesting_depth(name: &str) -> usize {
    name.matches("mcp__").count()
}

/// Backoff ceiling for downstream reconnect. `WOOLLAMA_MCP_RETRY_MAX_SECS`, default 60s.
/// **`0` disables retry entirely** — for a deployment that prefers a downstream to stay down
/// until someone looks at it.
fn retry_max() -> u64 {
    std::env::var("WOOLLAMA_MCP_RETRY_MAX_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(60)
}

/// Per-server connect timeout (handshake + initial tools/list). `WOOLLAMA_MCP_CONNECT_TIMEOUT_SECS`,
/// default 30s. Bounds startup so a hung downstream server can't wedge the daemon.
fn connect_timeout() -> std::time::Duration {
    let secs = std::env::var("WOOLLAMA_MCP_CONNECT_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30);
    std::time::Duration::from_secs(secs)
}

/// Map (server, tool) to a wire-safe tool name: `mcp__<server>__<tool>` — the same scheme
/// claude-code uses, and valid OpenAI/MCP function-name grammar (`[A-Za-z0-9_-]{1,64}`).
/// A dotted `server.tool` is rejected by strict OpenAI-compatible inferencers; a name that
/// would exceed 64 chars falls back to a deterministic hash (resolved via the reverse map).
fn wire_name(server: &str, tool: &str) -> String {
    let full = format!("mcp__{server}__{tool}");
    if full.len() <= 64 {
        full
    } else {
        format!("mcp__{:016x}", fnv1a64(full.as_bytes()))
    }
}

/// FNV-1a, 64-bit. Pinned deliberately: this hash names a tool **on the wire**, so it must be
/// reproducible across Rust versions, machines and processes.
///
/// It previously used `std::collections::hash_map::DefaultHasher`, whose algorithm std explicitly
/// does not guarantee between releases — so a `rustup update` could silently rename a tool, and a
/// renamed tool is one that stops resolving: a recipe's `tools` allow-list entry no longer
/// matches, and a client that cached the advertised name calls something that no longer exists.
/// The failure is silent on both sides.
///
/// Dormant while almost nothing exceeded 64 characters. Federation is what pushes names onto this
/// path — a consumed router's tools are already `mcp__X__Y`, so one hop of nesting adds ~12
/// characters — and in a mutual topology names gain a level per refresh. At that point a hashed
/// name stops being a cosmetic fallback and becomes a load-bearing identity.
///
/// FNV-1a rather than a crypto hash: this is a naming function, not a security boundary, and it
/// needs no dependency. Collisions resolve through `wire_index` the same way any name does.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Per-call timeout for a downstream tool invocation. `WOOLLAMA_MCP_CALL_TIMEOUT_SECS`,
/// default 120s. A hung downstream tool fails the one request instead of hanging it (and
/// leaking the request task) forever.
fn call_timeout() -> std::time::Duration {
    let secs = std::env::var("WOOLLAMA_MCP_CALL_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(120);
    std::time::Duration::from_secs(secs)
}

/// Invoke a downstream tool with the per-call timeout applied.
async fn call_with_timeout(peer: &Peer<RoleClient>, params: CallToolRequestParams) -> Result<CallToolResult, String> {
    let dur = call_timeout();
    match tokio::time::timeout(dur, peer.call_tool(params)).await {
        Ok(Ok(res)) => Ok(res),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err(format!("downstream tool call timed out after {}s", dur.as_secs())),
    }
}

impl McpRegistry {
    /// Connect to every configured server CONCURRENTLY, each bounded by a per-server timeout
    /// (best-effort: a server that fails to start OR hangs on the handshake is logged and
    /// skipped, so a single bad/slow server can neither take the router down nor block its
    /// startup). The timeout is `WOOLLAMA_MCP_CONNECT_TIMEOUT_SECS` (default 30s).
    /// How to report a server that didn't come up: `Failed` (terminal) when the transport itself
    /// can't be built, or when retry is switched off entirely — in both cases nothing will ever
    /// try again, and reporting `Retrying` would be the exact conflation `ServerHealth` exists to
    /// prevent. `Retrying` only when something really will retry.
    fn initial_health(spec: &McpServerSpec, attempts: u32, last_error: String) -> ServerHealth {
        if let Some(reason) = Self::transport_fault(spec) {
            return ServerHealth::Failed { reason };
        }
        if retry_max() == 0 {
            return ServerHealth::Failed { reason: format!("{last_error} (retry disabled)") };
        }
        ServerHealth::Retrying { attempts, last_error }
    }

    pub async fn connect(specs: HashMap<String, McpServerSpec>) -> McpRegistry {
        let specs_by_name = specs.clone();
        let timeout = connect_timeout();
        let results = futures::future::join_all(
            specs
                .into_iter()
                .map(|(name, spec)| async move {
                    let transport = spec.transport_name();
                    (name, tokio::time::timeout(timeout, Self::connect_one(&spec)).await, transport)
                }),
        )
        .await;
        let mut servers = HashMap::new();
        let mut health = HashMap::new();
        for (name, res, transport) in results {
            match res {
                Ok(Ok(conn)) => {
                    Self::capped_count(&name, &conn);
                    servers.insert(name.clone(), conn);
                    health.insert(name, (ServerHealth::Connected, transport));
                }
                Ok(Err(e)) => {
                    eprintln!("woollamad: MCP server '{name}' failed to start, skipping: {e}");
                    let h = Self::initial_health(&specs_by_name[&name], 1, e);
                    health.insert(name, (h, transport));
                }
                Err(_) => {
                    let e = format!("timed out after {}s connecting", timeout.as_secs());
                    eprintln!("woollamad: MCP server '{name}' {e}, skipping");
                    let h = Self::initial_health(&specs_by_name[&name], 1, e);
                    health.insert(name, (h, transport));
                }
            }
        }
        McpRegistry { inner: std::sync::RwLock::new(Arc::new(Snapshot::new(servers, health))) }
    }

    /// Swap in a snapshot with `name` connected. Rebuilds the wire index from the new server
    /// set, so an advertised name and the peer it resolves to are always published together.
    fn publish_connected(&self, name: String, conn: ServerConn, transport: &'static str) {
        // Once, here — not per request. The condition is static for a given roster.
        Self::capped_count(&name, &conn);
        let mut guard = self.inner.write().expect("registry lock poisoned");
        let mut servers = HashMap::new();
        // ServerConn isn't Clone (it owns a peer handle), so rebuild by moving out of the old
        // snapshot where we can and cloning the cheap Peer handle where we can't.
        for (k, v) in guard.servers.iter() {
            servers.insert(k.clone(), ServerConn { peer: v.peer.clone(), tools: v.tools.clone() });
        }
        servers.insert(name.clone(), conn);
        let mut health = guard.health.clone();
        health.insert(name, (ServerHealth::Connected, transport));
        *guard = Arc::new(Snapshot::new(servers, health));
    }

    /// Record a failed attempt without touching the connected set.
    fn mark_retrying(&self, name: &str, attempts: u32, last_error: String, transport: &'static str) {
        let mut guard = self.inner.write().expect("registry lock poisoned");
        let mut servers = HashMap::new();
        for (k, v) in guard.servers.iter() {
            servers.insert(k.clone(), ServerConn { peer: v.peer.clone(), tools: v.tools.clone() });
        }
        let mut health = guard.health.clone();
        health.insert(name.to_string(), (ServerHealth::Retrying { attempts, last_error }, transport));
        *guard = Arc::new(Snapshot::new(servers, health));
    }

    /// Every configured server with its current health and tool count — the runtime counterpart
    /// to `check-config`. A `Retrying` server appears here WITH its last error, rather than being
    /// silently absent: that distinction is the whole point (see [`ServerHealth`]).
    pub fn status(&self) -> Vec<ServerStatus> {
        self.introspect().1
    }

    /// Tool listing and per-server status derived from ONE snapshot.
    ///
    /// Two separate reads would let a reconnect land between them and produce an internally
    /// inconsistent answer — a server reported `retrying` with 0 tools while its tools already
    /// appear in the listing, or the reverse. Carrying health in the snapshot is pointless if
    /// callers then read it separately.
    pub fn introspect(&self) -> (Vec<(String, String, String)>, Vec<ServerStatus>) {
        let snap = self.snapshot();
        let exported = Self::exported_from(&snap);

        // `tools` is what this server CONTRIBUTES, i.e. post-cap — not its ingested roster.
        // Counting ingested tools would report a server as contributing tools that
        // `/v1/tools` does not list, contradicting the field's own meaning.
        let mut contributed: HashMap<&str, usize> = HashMap::new();
        for (server, _) in &exported {
            *contributed.entry(server.as_str()).or_default() += 1;
        }

        let listing = exported
            .iter()
            .map(|(server, t)| (server.clone(), t.name.to_string(), wire_name(server, &t.name)))
            .collect();

        let mut status: Vec<ServerStatus> = snap
            .health
            .iter()
            .map(|(name, (health, transport))| ServerStatus {
                name: name.clone(),
                transport,
                health: health.clone(),
                tools: contributed.get(name.as_str()).copied().unwrap_or(0),
            })
            .collect();
        status.sort_by(|a, b| a.name.cmp(&b.name));
        (listing, status)
    }

    /// True once every configured server is `Connected` — used by tests to await convergence
    /// rather than sleeping a fixed duration.
    pub fn all_connected(&self) -> bool {
        self.snapshot().health.values().all(|(h, _)| *h == ServerHealth::Connected)
    }

    /// The current view. Cloning the `Arc` under the guard keeps the critical section to a
    /// pointer copy, so callers can `await` freely against a stable snapshot.
    fn snapshot(&self) -> Arc<Snapshot> {
        self.inner.read().expect("registry lock poisoned").clone()
    }

    /// Could this spec ever connect, or is it broken until a human edits the file?
    /// A transport that cannot be BUILT (an unusable header name/value, an HTTP client that will
    /// not initialize) is a config fault: retrying it forever only produces noise.
    fn transport_fault(spec: &McpServerSpec) -> Option<String> {
        match spec {
            McpServerSpec::Stdio(_) => None,
            McpServerSpec::Http(h) => http_transport(h).err(),
        }
    }

    async fn connect_one(spec: &McpServerSpec) -> Result<ServerConn, String> {
        // Only the transport construction is variant-specific; everything from here on —
        // peer handling, list_all_tools, the wire_index — is transport-agnostic.
        let running = match spec {
            McpServerSpec::Stdio(s) => {
                let mut cmd = tokio::process::Command::new(&s.command);
                // Scrub the child env: a downstream tool server must NOT inherit the daemon's
                // provider secrets (ANTHROPIC_API_KEY etc.). Mirrors the claude-code child scrub
                // and the Python MCP SDK's default-scrubbed stdio environment.
                cmd.args(&s.args).env_clear().envs(merged_env(&s.env));
                let transport = TokioChildProcess::new(cmd).map_err(|e| e.to_string())?;
                ().serve(transport).await.map_err(|e| e.to_string())?
            }
            McpServerSpec::Http(h) => {
                // No child process, so no env scrub applies. Everything after `.serve()` is
                // transport-agnostic and shared with the stdio path.
                let transport = http_transport(h)?;
                ().serve(transport).await.map_err(|e| e.to_string())?
            }
        };
        let peer = running.peer().clone();
        let tools = peer.list_all_tools().await.map_err(|e| e.to_string())?;
        // Keep the connection alive for the process lifetime: dropping the
        // RunningService would cancel its task and close the child. The router holds
        // these for as long as it runs, so leaking the handle is intentional here.
        // (A graceful-shutdown lifecycle can replace this later.)
        std::mem::forget(running);
        Ok(ServerConn { peer, tools })
    }

    /// Resolve an advertised wire name (`mcp__server__tool`) to (server peer, bare tool) via
    /// the reverse map built at connect — unambiguous, unlike splitting on a separator.
    fn resolve(&self, wire: &str) -> Option<(Peer<RoleClient>, String)> {
        let snap = self.snapshot();
        let (server, bare) = snap.wire_index.get(wire)?;
        let conn = snap.servers.get(server)?;
        Some((conn.peer.clone(), bare.clone()))
    }

    /// Returns an OWNED tool: the snapshot it came from may be replaced by a reconnect while
    /// the caller still holds it, so a borrow tied to `&self` would be a lifetime lie.
    fn tool(&self, namespaced: &str) -> Option<Tool> {
        let (server, bare) = namespaced.split_once('.')?;
        self.snapshot().servers.get(server)?.tools.iter().find(|t| t.name == bare).cloned()
    }

    /// Every downstream tool that survives the federation nesting cap, as
    /// `(server, original tool)`. The single place the cap is applied, so what a client is
    /// TOLD about and what gets re-exported can never disagree.
    fn exported(&self) -> Vec<(String, Tool)> {
        Self::exported_from(&self.snapshot())
    }

    /// Derive the exported set from ONE snapshot, so callers that need tools and health together
    /// read them from the same instant.
    ///
    /// Deliberately silent: this runs on the request path (per MCP `tools/list`, per
    /// `GET /v1/tools`), and the capped condition is static, so logging here would emit one line
    /// per request forever. The cap is reported once at ingest instead.
    fn exported_from(snap: &Snapshot) -> Vec<(String, Tool)> {
        let cap = max_nesting();
        let mut out = Vec::new();
        for (server, conn) in &snap.servers {
            for t in &conn.tools {
                // Re-exporting adds one level, so compare the RESULTING depth against the cap.
                if cap > 0 && nesting_depth(&t.name) + 1 > cap {
                    continue;
                }
                out.push((server.clone(), t.clone()));
            }
        }
        // Stable order: `snap.servers` is a HashMap, so without this the MCP tools/list roster and
        // `/v1/tools` come back shuffled on every call — which also churns any upstream cache
        // keyed on the tool list. `status()` sorts for the same reason.
        out.sort_by(|a, b| (&a.0, &a.1.name).cmp(&(&b.0, &b.1.name)));
        out
    }

    /// How many of a server's ingested tools the nesting cap drops. Reported once at ingest.
    fn capped_count(server: &str, conn: &ServerConn) -> usize {
        let cap = max_nesting();
        if cap == 0 {
            return 0;
        }
        let n = conn.tools.iter().filter(|t| nesting_depth(&t.name) + 1 > cap).count();
        if n > 0 {
            eprintln!(
                "woollamad: MCP server '{server}': {n} tool(s) already at {cap} levels of \
                 federation namespacing will not be re-exported (WOOLLAMA_MCP_MAX_NESTING={cap})"
            );
        }
        n
    }

    /// Every downstream tool, re-exported namespaced with input + output schema MIRRORED —
    /// for woollama's own tools/list (the MCP aggregator).
    pub fn reexport_tools(&self) -> Vec<Tool> {
        self.exported()
            .into_iter()
            .map(|(server, t)| {
                let mut nt = Tool::new(
                    wire_name(&server, &t.name),
                    t.description.clone().unwrap_or_default(),
                    t.input_schema.clone(),
                );
                if let Some(os) = t.output_schema.clone() {
                    nt = nt.with_raw_output_schema(os);
                }
                nt
            })
            .collect()
    }

    /// `(server, bare name, advertised wire name)` for every re-exported tool — the introspection
    /// view behind `GET /v1/tools`. Derived from the same `exported()` set as the aggregator, so
    /// introspection cannot advertise a tool the aggregator drops (or vice versa).
    pub fn tool_listing(&self) -> Vec<(String, String, String)> {
        self.introspect().0
    }

    /// Call a tool by BARE name on a specific server (for the MCP conversation-store
    /// provider, whose tools — create_thread/etc. — aren't recipe-namespaced).
    pub async fn call_server(&self, server: &str, tool: &str, args: &Value) -> Result<CallToolResult, String> {
        let peer = {
            let snap = self.snapshot();
            snap.servers.get(server).ok_or_else(|| format!("unknown server '{server}'"))?.peer.clone()
        };
        let mut params = CallToolRequestParams::new(tool.to_string());
        if let Some(obj) = args.as_object() {
            params = params.with_arguments(obj.clone());
        }
        call_with_timeout(&peer, params).await
    }

    /// Dispatch a namespaced tool and return the RAW `CallToolResult` (content +
    /// structured_content), for the MCP proxy passthrough (vs. the lossy text render
    /// `RegistryToolProvider::dispatch` does for the inference loop).
    pub async fn call_raw(&self, namespaced: &str, args: &Value) -> Result<CallToolResult, String> {
        let Some((peer, bare)) = self.resolve(namespaced) else {
            return Err(format!("unknown tool '{namespaced}'"));
        };
        let mut params = CallToolRequestParams::new(bare);
        if let Some(obj) = args.as_object() {
            params = params.with_arguments(obj.clone());
        }
        call_with_timeout(&peer, params).await
    }
}

/// Adapts an `McpRegistry` to the engine's `ToolProvider` seam.
pub struct RegistryToolProvider {
    pub reg: Arc<McpRegistry>,
}

#[async_trait::async_trait]
impl ToolProvider for RegistryToolProvider {
    fn tool_schemas(&self, allow: &[String]) -> Result<Vec<Value>, EngineError> {
        let mut out = Vec::new();
        for namespaced in allow {
            let Some(tool) = self.reg.tool(namespaced) else {
                eprintln!("woollamad: recipe references unknown tool '{namespaced}', skipping");
                continue;
            };
            // Recipe config is human-friendly `server.tool`; advertise the wire-safe
            // `mcp__server__tool` so the model emits a name we resolve via the reverse map.
            let Some((server, bare)) = namespaced.split_once('.') else { continue };
            out.push(json!({
                "type": "function",
                "function": {
                    "name": wire_name(server, bare),
                    "description": tool.description.as_deref().unwrap_or(""),
                    "parameters": Value::Object((*tool.input_schema).clone()),
                },
            }));
        }
        Ok(out)
    }

    async fn dispatch(&self, name: &str, args: &Value) -> (String, bool) {
        let Some((peer, bare)) = self.reg.resolve(name) else {
            return (format!("ERROR: unknown tool '{name}'"), false);
        };
        let mut params = CallToolRequestParams::new(bare);
        if let Some(obj) = args.as_object() {
            params = params.with_arguments(obj.clone());
        }
        match call_with_timeout(&peer, params).await {
            Ok(res) => render_result(&res),
            Err(e) => (format!("ERROR: {e}"), false),
        }
    }
}

/// Render a downstream `CallToolResult` to the `(content, ok)` the loop feeds back:
/// joined text blocks, else the structured payload as JSON; `is_error` → `ok=false`.
fn render_result(res: &CallToolResult) -> (String, bool) {
    let is_error = res.is_error.unwrap_or(false);
    let text: Vec<String> =
        res.content.iter().filter_map(|c| c.as_text().map(|t| t.text.clone())).collect();
    let mut body = if !text.is_empty() {
        text.join("\n")
    } else if let Some(sc) = &res.structured_content {
        serde_json::to_string(sc).unwrap_or_default()
    } else {
        String::new()
    };
    if is_error {
        body = if body.is_empty() { "[tool error]".to_string() } else { format!("[tool error] {body}") };
    }
    (body, !is_error)
}


/// Retry every downstream that isn't connected, on a per-server exponential backoff (1s doubling
/// to [`retry_max`]). One task per server; each exits as soon as its server connects.
///
/// **Refresh is background-only and never request-triggered.** That is deliberate and
/// load-bearing: `list_tools` serves the cached snapshot, so a request can never cause a
/// downstream fetch. If it could, then in a federated topology A's `tools/list` would trigger a
/// fetch from B, whose `tools/list` would fetch from A — live recursion across routers. Keeping
/// refresh on a timer preserves the property that makes federation safe today.
///
/// A `Failed` server (its transport could not be built) is NOT retried: that is a config fault,
/// and retrying would produce noise until someone edits a file.
pub fn spawn_reconnect(reg: Arc<McpRegistry>, specs: HashMap<String, McpServerSpec>) {
    let max = retry_max();
    if max == 0 {
        return; // retry disabled by configuration
    }
    for (name, spec) in specs {
        let health = reg.snapshot().health.get(&name).map(|(h, _)| h.clone());
        if !matches!(health, Some(ServerHealth::Retrying { .. })) {
            continue; // already connected, or terminally Failed
        }
        let reg = reg.clone();
        tokio::spawn(async move {
            let transport = spec.transport_name();
            let mut attempts = 1u32;
            let mut delay = 1u64;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                delay = (delay * 2).min(max);
                attempts += 1;
                match tokio::time::timeout(connect_timeout(), McpRegistry::connect_one(&spec)).await {
                    Ok(Ok(conn)) => {
                        eprintln!("woollamad: MCP server '{name}' reconnected after {attempts} attempt(s)");
                        reg.publish_connected(name, conn, transport);
                        return;
                    }
                    Ok(Err(e)) => reg.mark_retrying(&name, attempts, e, transport),
                    Err(_) => reg.mark_retrying(&name, attempts, "connect timed out".to_string(), transport),
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HttpSpec, StdioSpec};

    #[test]
    #[cfg(unix)]
    fn scrubbed_env_excludes_provider_secrets() {
        std::env::set_var("ANTHROPIC_API_KEY", "leak-me");
        std::env::set_var("OPENAI_API_KEY", "leak-me-2");
        let env = scrubbed_env();
        assert!(!env.contains_key("ANTHROPIC_API_KEY"), "provider key must not reach MCP servers");
        assert!(!env.contains_key("OPENAI_API_KEY"));
        if std::env::var_os("PATH").is_some() {
            assert!(env.contains_key("PATH"), "PATH must survive so the server interpreter resolves");
        }
        for k in env.keys() {
            assert!(
                crate::claude_code::CHILD_ENV_ALLOW.contains(&k.as_str()) || k.starts_with("LC_"),
                "leaked non-allow-listed var to an MCP server: {k}"
            );
        }
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn hung_server_does_not_block_startup() {
        std::env::set_var("WOOLLAMA_MCP_CONNECT_TIMEOUT_SECS", "1");
        let mut specs = HashMap::new();
        // `sleep` spawns but never speaks MCP -> the initialize handshake hangs -> timed out.
        specs.insert(
            "hung".to_string(),
            McpServerSpec::Stdio(StdioSpec { command: "sleep".into(), args: vec!["30".into()], env: HashMap::new() }),
        );
        // `false` exits immediately -> connect_one errors -> skipped.
        specs.insert(
            "dead".to_string(),
            McpServerSpec::Stdio(StdioSpec { command: "false".into(), args: vec![], env: HashMap::new() }),
        );
        let start = std::time::Instant::now();
        let reg = McpRegistry::connect(specs).await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "a hung downstream server must not block startup (took {elapsed:?})"
        );
        assert!(reg.snapshot().servers.is_empty(), "hung + dead servers are skipped, not registered");
        std::env::remove_var("WOOLLAMA_MCP_CONNECT_TIMEOUT_SECS");
    }

    #[test]
    #[cfg(unix)]
    fn spec_env_merges_over_the_scrub_without_reopening_it() {
        std::env::set_var("ANTHROPIC_API_KEY", "leak-me");
        let mut spec_env = HashMap::new();
        spec_env.insert("GIT_AUTHOR_NAME".to_string(), "woollama".to_string());
        let merged = merged_env(&spec_env);
        // The half that makes the feature work.
        assert_eq!(merged.get("GIT_AUTHOR_NAME").map(String::as_str), Some("woollama"));
        // The half that rots if nobody pins it: the scrub is still a floor. Nothing reaches a
        // tool server by inheritance just because the spec has an `env` block.
        assert!(
            !merged.contains_key("ANTHROPIC_API_KEY"),
            "a non-allow-listed var must not reach the server just because `env` exists"
        );
    }

    #[test]
    #[cfg(unix)]
    fn spec_env_can_deliberately_reinject_a_provider_key() {
        let mut spec_env = HashMap::new();
        spec_env.insert("ANTHROPIC_API_KEY".to_string(), "on-purpose".to_string());
        // Explicit, in the operator's config, greppable — categorically different from
        // inheriting one silently. Pinned so nobody later "hardens" it into a surprise.
        assert_eq!(
            merged_env(&spec_env).get("ANTHROPIC_API_KEY").map(String::as_str),
            Some("on-purpose")
        );
    }

    #[tokio::test]
    async fn a_dead_downstream_reports_retrying_with_its_error_not_absence() {
        // The whole point of ServerHealth. A router that reported an unreachable-but-retryable
        // server as simply *missing* would look healthy with its tools quietly gone — which is
        // the failure shape this project keeps finding. Absence is not a status.
        let mut specs = HashMap::new();
        specs.insert(
            "gone".to_string(),
            McpServerSpec::Http(HttpSpec { url: "http://127.0.0.1:1/mcp".into(), headers: HashMap::new() }),
        );
        specs.insert(
            "dead".to_string(),
            McpServerSpec::Stdio(StdioSpec { command: "false".into(), args: vec![], env: HashMap::new() }),
        );
        let reg = McpRegistry::connect(specs).await;

        let status = reg.status();
        assert_eq!(status.len(), 2, "every CONFIGURED server must be reported, not just live ones: {status:?}");
        for s in &status {
            match &s.health {
                ServerHealth::Retrying { last_error, .. } => {
                    assert!(!last_error.is_empty(), "a retrying server must carry WHY: {s:?}")
                }
                other => panic!("expected Retrying for '{}', got {other:?}", s.name),
            }
            assert_eq!(s.tools, 0, "a server that isn't connected contributes no tools");
        }
        assert_eq!(status[0].name, "dead", "status is sorted by name for stable operator output");
        assert_eq!(status[0].transport, "stdio");
        assert_eq!(status[1].transport, "http");
    }

    #[tokio::test]
    async fn an_unbuildable_transport_is_terminal_not_retried_forever() {
        // `Failed` existed but was never CONSTRUCTED: every failure folded into `Retrying`, so a
        // config fault — an unusable header, say — spawned a retry task that hammered a broken
        // spec forever at the backoff ceiling, contradicting the docs, the enum's own doc comment,
        // and spawn_reconnect's "terminally Failed" branch.
        let mut headers = HashMap::new();
        headers.insert("Invalid Header Name".to_string(), "x".to_string());
        let mut specs = HashMap::new();
        specs.insert(
            "broken".to_string(),
            McpServerSpec::Http(HttpSpec { url: "http://127.0.0.1:1/mcp".into(), headers }),
        );
        let reg = McpRegistry::connect(specs).await;
        match &reg.status()[0].health {
            ServerHealth::Failed { reason } => {
                assert!(!reason.is_empty(), "a terminal failure must say why")
            }
            other => panic!("a transport that cannot be built is terminal, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_transport_that_cannot_be_built_is_skipped_not_fatal() {
        // `http_transport` has two construction failures that are not connection failures:
        // an unusable header, and `reqwest::Client::builder().build()` erroring when TLS or the
        // system resolver can't initialize. The latter is not reproducible in-process (it needs a
        // broken CA store or /etc/resolv.conf), but both return through the SAME `?` path, so
        // this pins that the path degrades to a skipped server rather than taking the daemon
        // down — connect_one runs unspawned inside build_state, where a panic would reach main.
        let mut headers = HashMap::new();
        headers.insert("Invalid Header Name".to_string(), "x".to_string());
        let mut specs = HashMap::new();
        specs.insert(
            "bad".to_string(),
            McpServerSpec::Http(HttpSpec { url: "http://127.0.0.1:1/mcp".into(), headers }),
        );
        let reg = McpRegistry::connect(specs).await;
        assert!(reg.snapshot().servers.is_empty(), "an unbuildable transport is skipped, not registered");
    }

    #[test]
    fn nesting_depth_counts_federation_levels() {
        assert_eq!(nesting_depth("count_to"), 0, "a plain downstream tool");
        assert_eq!(nesting_depth("mcp__fix__count_to"), 1, "one hop of federation");
        assert_eq!(nesting_depth("mcp__b__mcp__a__count_to"), 2, "two hops");
    }

    #[test]
    fn hashed_wire_names_are_pinned_across_toolchains() {
        // LITERAL expected values, deliberately. This hash names a tool ON THE WIRE, so the
        // property that matters is not "it hashes" but "it hashes to the same thing it did
        // yesterday" — which a property-based test cannot express. The previous implementation
        // used std's DefaultHasher, whose algorithm is explicitly not guaranteed between
        // releases, so a toolchain upgrade could silently rename a tool. These assertions fail
        // instead.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325, "FNV-1a offset basis");
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8, "published FNV-1a test vector");

        // And the same, end to end through the naming function.
        let long = wire_name(&"s".repeat(50), &"t".repeat(50));
        assert_eq!(long, "mcp__c251c716388ed55d", "an over-long name must hash to a STABLE form");
    }

    #[test]
    fn wire_name_is_valid_and_namespaced() {
        let w = wire_name("hello", "count_to");
        assert_eq!(w, "mcp__hello__count_to");
        let ok = |s: &str| s.len() <= 64 && !s.is_empty()
            && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        assert!(ok(&w), "must satisfy the OpenAI/MCP function-name grammar (no dots, <=64)");
        // An overlong combination falls back to a hashed, still-valid name.
        let long = wire_name(&"s".repeat(50), &"t".repeat(50));
        assert!(ok(&long) && long.starts_with("mcp__"), "overlong name must hash to a valid form");
    }

    #[test]
    fn call_timeout_honors_env_and_default() {
        std::env::remove_var("WOOLLAMA_MCP_CALL_TIMEOUT_SECS");
        assert_eq!(call_timeout().as_secs(), 120, "default per-call timeout");
        std::env::set_var("WOOLLAMA_MCP_CALL_TIMEOUT_SECS", "7");
        assert_eq!(call_timeout().as_secs(), 7, "env override");
        std::env::remove_var("WOOLLAMA_MCP_CALL_TIMEOUT_SECS");
    }
}
