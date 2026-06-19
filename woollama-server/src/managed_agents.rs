//! Anthropic Managed Agents — the `managed-agents` conversation backend (conv-6),
//! ported (woollama-side) from Python `woollama.managed_agents`.
//!
//! Anthropic owns the session/transcript/loop/container; woollama holds only the
//! session_id. Auth is `ANTHROPIC_API_KEY` (paid, distinct from the keyless
//! claude-resume/claude-code subscription path). One tool-less agent per model is
//! created lazily + cached (an agent is a reusable account object — never per session).
//!
//! WIRE FORMAT (reconciled against the official `anthropic` SDK 0.109 + LIVE-VERIFIED against
//! the real API 2026-06-18 — Managed Agents is **event-driven + SSE-streamed**, NOT a
//! synchronous turn endpoint):
//!
//! - create env+agent (lazy, cached): `POST /v1/environments {name, config:{type:"cloud", networking:{type:"unrestricted"}}}` then `POST /v1/agents {name, model, tools}`.
//! - create session: `POST /v1/sessions {agent, environment_id}` → `{id}` (`agent` accepts the bare id string).
//! - send a turn: `POST /v1/sessions/{id}/events?beta=true` with `{events:[{type:"user.message", content:[{type:"text",text}]}]}`.
//! - read the answer: `GET /v1/sessions/{id}/events/stream?beta=true` (the DEDICATED stream route, distinct from the list route) with `accept: text/event-stream` → SSE.
//! - the stream live-tails the CURRENT turn from its start (it replays the just-posted `user.message`, so send→stream is race-free) and does NOT replay prior turns (verified: turn-2 carried only fresh event ids).
//! - assistant text arrives as `agent.message` events (`content:[{type:"text",text}]`); the turn ends at `session.status_idle` whose `stop_reason.type` is `end_turn`, or `requires_action` (with `stop_reason.event_ids` naming the blocking `agent.custom_tool_use` ids) when the agent paused on a custom tool.
//! - other events (status_running, thinking, spans, thread_status_*) are ignored.
//! - resume (ask_user): `POST …/events?beta=true` with `{events:[{type:"user.custom_tool_result", custom_tool_use_id, content:[{type:"text",text}]}]}`.
//! - history: `GET /v1/sessions/{id}/events?beta=true` (the LIST route) → `{data:[...all events...]}`.
//! - teardown: `DELETE /v1/sessions/{id}?beta=true` (agents archive via `POST …/archive`, not delete).
//!
//! Docs: platform.claude.com/docs/en/managed-agents/{overview,sessions,events-and-streaming}.
//!
//! ✅ CONFIDENCE: LIVE-VERIFIED 2026-06-18 against api.anthropic.com (haiku, 2-turn recall +
//! history + teardown). The reconciliation found one real bug — the stream route is
//! `…/events/stream`, NOT `…/events` — now fixed. The remaining live-untested path is the
//! `requires_action`/`ask_user` pause-resume branch (the probe didn't trigger a custom-tool
//! call); its shapes match the SDK types but the round-trip isn't yet exercised end-to-end.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::Mutex;

const DEFAULT_MODEL: &str = "claude-opus-4-8";
const BETA_HEADER: &str = "managed-agents-2026-04-01";

#[derive(Debug)]
pub struct ManagedAgentsError(pub String);

impl std::fmt::Display for ManagedAgentsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The result of a turn: assistant text, plus `pending` when the agent paused on the
/// client-side `ask_user` custom tool (the interactive requires_action signal).
pub struct Turn {
    pub text: String,
    pub pending: Option<Pending>,
}

pub struct Pending {
    /// The `agent.custom_tool_use` event id — passed back as `custom_tool_use_id` on resume.
    pub id: String,
    pub input: Value,
}

/// Map `claude-agent/<model>` to a full Anthropic model id (opus/sonnet/haiku expand;
/// a full id passes through; empty → default).
pub fn resolve_model(model: &str) -> String {
    let name = model.split_once('/').map(|(_, m)| m).unwrap_or(model);
    match name {
        "" => DEFAULT_MODEL.to_string(),
        "opus" => "claude-opus-4-8".to_string(),
        "sonnet" => "claude-sonnet-4-6".to_string(),
        "haiku" => "claude-haiku-4-5".to_string(),
        other => other.to_string(),
    }
}

/// The managed-agents backend: holds the lazily-created, reused environment + per-model
/// agents (created once, never per session — the documented anti-pattern). When
/// `persist_path` is `Some` (opt-in via `WOOLLAMA_MANAGED_AGENTS_PERSIST`), the env + agent
/// ids are written there and REUSED across daemon restarts — verified-on-first-use, so a
/// since-archived id self-heals by recreating instead of bricking the path. When `None`
/// (default), the ids live only for this process (each restart creates its own).
pub struct ManagedAgents {
    base_url: String,
    persist_path: Option<PathBuf>,
    state: Mutex<Setup>,
}

#[derive(Default)]
struct Setup {
    env_id: Option<String>,
    agents: HashMap<String, String>, // full model id → agent_id
    loaded: bool,                     // have we seeded from persist_path yet
    env_verified: bool,               // confirmed the persisted env_id still exists this process
    agents_verified: HashSet<String>, // model keys whose persisted agent_id we've confirmed
}

impl ManagedAgents {
    pub fn new(persist_path: Option<PathBuf>) -> Self {
        let base_url = std::env::var("ANTHROPIC_BASE_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());
        Self::with_base_url(base_url, persist_path)
    }

    #[doc(hidden)]
    pub fn with_base_url(base_url: String, persist_path: Option<PathBuf>) -> Self {
        ManagedAgents {
            base_url: base_url.trim_end_matches('/').to_string(),
            persist_path,
            state: Mutex::new(Setup::default()),
        }
    }

    fn request(&self, method: reqwest::Method, path: &str) -> Result<reqwest::RequestBuilder, ManagedAgentsError> {
        let key = std::env::var("ANTHROPIC_API_KEY").ok().filter(|s| !s.is_empty()).ok_or_else(|| {
            ManagedAgentsError(
                "ANTHROPIC_API_KEY is not set — the managed-agents backend is a paid, \
                 key-authenticated path (distinct from the keyless claude-resume backend)."
                    .to_string(),
            )
        })?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(330))
            .build()
            .map_err(|e| ManagedAgentsError(e.to_string()))?;
        Ok(client
            .request(method, format!("{}{path}", self.base_url))
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", BETA_HEADER))
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value, ManagedAgentsError> {
        let r = self
            .request(reqwest::Method::POST, path)?
            .json(&body)
            .send()
            .await
            .map_err(|e| ManagedAgentsError(format!("managed-agents API error: {e}")))?;
        if !r.status().is_success() {
            return Err(ManagedAgentsError(format!("managed-agents API status {} on POST {path}", r.status())));
        }
        r.json().await.map_err(|e| ManagedAgentsError(format!("managed-agents bad json: {e}")))
    }

    /// `GET` an object by id to confirm it still exists server-side (200 vs 404). Used to
    /// validate a PERSISTED id on first use so a since-archived env/agent self-heals.
    async fn exists(&self, prefix: &str, id: &str) -> bool {
        match self.request(reqwest::Method::GET, &format!("{prefix}{id}?beta=true")) {
            Ok(rb) => rb.send().await.map(|r| r.status().is_success()).unwrap_or(false),
            Err(_) => false,
        }
    }

    /// Write the current env + agent ids to `persist_path` (best-effort; a write failure only
    /// costs cross-restart reuse, never correctness).
    fn persist(&self, s: &Setup) {
        if let Some(p) = &self.persist_path {
            let body = json!({"env_id": s.env_id, "agents": s.agents});
            if let Err(e) = std::fs::write(p, body.to_string()) {
                eprintln!("woollamad: managed-agents persist write failed ({}): {e}", p.display());
            }
        }
    }

    /// Lazily create + cache the shared environment and a per-model agent. With persistence
    /// enabled, seed from disk once, verify a reused id on first use, and recreate + rewrite
    /// only when absent or gone.
    async fn ensure_agent(&self, model: &str) -> Result<(String, String), ManagedAgentsError> {
        let full = resolve_model(model);
        let mut s = self.state.lock().await;

        // Seed the cache from the persisted file once (opt-in; absent file = empty seed).
        if !s.loaded {
            if let Some(p) = &self.persist_path {
                if let Ok(bytes) = std::fs::read(p) {
                    if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
                        s.env_id = v.get("env_id").and_then(Value::as_str).map(String::from);
                        if let Some(map) = v.get("agents").and_then(Value::as_object) {
                            s.agents = map
                                .iter()
                                .filter_map(|(k, val)| val.as_str().map(|x| (k.clone(), x.to_string())))
                                .collect();
                        }
                    }
                }
            }
            s.loaded = true;
        }

        let mut changed = false;

        // Environment: verify a persisted id on first use; drop it if the server no longer has it.
        if let Some(env) = s.env_id.clone() {
            if self.persist_path.is_some() && !s.env_verified && !self.exists("/v1/environments/", &env).await {
                s.env_id = None;
            }
            s.env_verified = true;
        }
        if s.env_id.is_none() {
            let env = self.post("/v1/environments", json!({
                "name": "woollama-agents",
                "config": {"type": "cloud", "networking": {"type": "unrestricted"}}
            })).await?;
            s.env_id = Some(env["id"].as_str().unwrap_or_default().to_string());
            changed = true;
        }

        // Per-model agent: same verify-then-create.
        if let Some(agent) = s.agents.get(&full).cloned() {
            if self.persist_path.is_some()
                && !s.agents_verified.contains(&full)
                && !self.exists("/v1/agents/", &agent).await
            {
                s.agents.remove(&full);
            }
            s.agents_verified.insert(full.clone());
        }
        if !s.agents.contains_key(&full) {
            let agent = self.post("/v1/agents", json!({
                "name": format!("woollama:{full}"),
                "model": full,
                "tools": [ask_user_tool()],
            })).await?;
            s.agents.insert(full.clone(), agent["id"].as_str().unwrap_or_default().to_string());
            changed = true;
        }

        if changed {
            self.persist(&s);
        }
        Ok((s.agents[&full].clone(), s.env_id.clone().unwrap()))
    }

    /// Create the hosted session (lazily, on the first turn). Returns its session_id.
    /// `title`/`metadata` are tracked woollama-side (the handle table); the documented
    /// session-create body is `{agent, environment_id}`, so we don't forward them.
    pub async fn create_session(&self, model: &str, _title: Option<&str>, _metadata: &Value) -> Result<String, ManagedAgentsError> {
        let (agent_id, env_id) = self.ensure_agent(model).await?;
        let s = self.post("/v1/sessions", json!({"agent": agent_id, "environment_id": env_id})).await?;
        Ok(s["id"].as_str().unwrap_or_default().to_string())
    }

    /// Send a single event to a session (POST …/events). The assistant's response is NOT in
    /// this reply — it streams over `stream_turn` (the event-driven model).
    async fn send_event(&self, session_id: &str, event: Value) -> Result<(), ManagedAgentsError> {
        let r = self
            .request(reqwest::Method::POST, &format!("/v1/sessions/{session_id}/events?beta=true"))?
            .json(&json!({"events": [event]}))
            .send()
            .await
            .map_err(|e| ManagedAgentsError(format!("managed-agents API error: {e}")))?;
        if !r.status().is_success() {
            return Err(ManagedAgentsError(format!("managed-agents API status {} sending event", r.status())));
        }
        Ok(())
    }

    /// Consume the session's SSE event stream until the turn ends (`session.status_idle`),
    /// collecting `agent.message` text and detecting a custom-tool pause.
    async fn stream_turn(&self, session_id: &str) -> Result<Turn, ManagedAgentsError> {
        let resp = self
            .request(reqwest::Method::GET, &format!("/v1/sessions/{session_id}/events/stream?beta=true"))?
            .header("accept", "text/event-stream")
            .send()
            .await
            .map_err(|e| ManagedAgentsError(format!("managed-agents API error: {e}")))?;
        if !resp.status().is_success() {
            return Err(ManagedAgentsError(format!("managed-agents API status {} on event stream", resp.status())));
        }

        let mut text = String::new();
        let mut tool_uses: HashMap<String, Value> = HashMap::new(); // custom_tool_use event id → input
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ManagedAgentsError(format!("managed-agents stream error: {e}")))?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            // SSE frames are separated by a blank line.
            while let Some(pos) = buf.find("\n\n") {
                let frame: String = buf[..pos].to_string();
                buf.drain(..pos + 2);
                let data: String = frame
                    .lines()
                    .filter_map(|l| l.strip_prefix("data:"))
                    .map(|d| d.trim_start())
                    .collect::<Vec<_>>()
                    .join("\n");
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let ev: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                match ev.get("type").and_then(Value::as_str) {
                    Some("agent.message") => text.push_str(&event_text(&ev)),
                    Some("agent.custom_tool_use") => {
                        if let Some(id) = ev.get("id").and_then(Value::as_str) {
                            tool_uses.insert(id.to_string(), ev.get("input").cloned().unwrap_or_else(|| json!({})));
                        }
                    }
                    Some("session.status_idle") => {
                        let kind = ev.get("stop_reason").and_then(|s| s.get("type")).and_then(Value::as_str);
                        let pending = if kind == Some("requires_action") {
                            ev.get("stop_reason")
                                .and_then(|s| s.get("event_ids"))
                                .and_then(Value::as_array)
                                .and_then(|ids| {
                                    ids.iter().filter_map(Value::as_str).find_map(|id| {
                                        tool_uses.get(id).map(|input| Pending { id: id.to_string(), input: input.clone() })
                                    })
                                })
                        } else {
                            None
                        };
                        return Ok(Turn { text, pending });
                    }
                    Some("session.error") => {
                        return Err(ManagedAgentsError(format!("managed-agents session error: {ev}")));
                    }
                    _ => {}
                }
            }
        }
        Err(ManagedAgentsError("managed-agents event stream ended without session.status_idle".to_string()))
    }

    /// One conversational turn (a new user message): send it, then stream the response.
    pub async fn run_turn(&self, session_id: &str, text: &str) -> Result<Turn, ManagedAgentsError> {
        self.send_event(session_id, json!({"type": "user.message", "content": [{"type": "text", "text": text}]})).await?;
        self.stream_turn(session_id).await
    }

    /// Resume a paused session by returning the user's answer to the pending custom tool.
    pub async fn answer_turn(&self, session_id: &str, custom_tool_use_id: &str, answer: &str) -> Result<Turn, ManagedAgentsError> {
        self.send_event(
            session_id,
            json!({"type": "user.custom_tool_result", "custom_tool_use_id": custom_tool_use_id, "content": [{"type": "text", "text": answer}]}),
        )
        .await?;
        self.stream_turn(session_id).await
    }

    /// Retrieve the transcript (Anthropic owns the bytes; woollama reshapes to messages).
    pub async fn history(&self, session_id: &str) -> Result<Vec<Value>, ManagedAgentsError> {
        let r = self
            .request(reqwest::Method::GET, &format!("/v1/sessions/{session_id}/events?beta=true"))?
            .send()
            .await
            .map_err(|e| ManagedAgentsError(format!("managed-agents API error: {e}")))?;
        if !r.status().is_success() {
            return Err(ManagedAgentsError(format!("managed-agents API status {} on events", r.status())));
        }
        let v: Value = r.json().await.map_err(|e| ManagedAgentsError(format!("managed-agents bad json: {e}")))?;
        Ok(events_to_messages(v.get("data").and_then(Value::as_array).cloned().unwrap_or_default()))
    }

    pub async fn delete_session(&self, session_id: &str) -> Result<(), ManagedAgentsError> {
        self.request(reqwest::Method::DELETE, &format!("/v1/sessions/{session_id}?beta=true"))?
            .send()
            .await
            .map_err(|e| ManagedAgentsError(format!("managed-agents API error: {e}")))?;
        Ok(())
    }
}

fn ask_user_tool() -> Value {
    json!({
        "type": "custom",
        "name": "ask_user",
        "description": "Ask the user a question and pause until they answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["question"]
        }
    })
}

/// Parse the event list into woollama transcript messages (user.message → user,
/// agent.message → assistant; tool/status events skipped for v1).
pub fn events_to_messages(events: Vec<Value>) -> Vec<Value> {
    let mut out = Vec::new();
    for e in events {
        match e.get("type").and_then(Value::as_str) {
            Some("user.message") => out.push(json!({"role": "user", "content": event_text(&e)})),
            Some("agent.message") => out.push(json!({"role": "assistant", "content": event_text(&e)})),
            _ => {}
        }
    }
    out
}

fn event_text(event: &Value) -> String {
    event
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<String>()
        })
        .unwrap_or_default()
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use axum::extract::Path as AxPath;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::{get, post};
    use axum::{Json, Router};

    #[derive(Default)]
    struct Counts {
        env: AtomicUsize,   // POST /v1/environments
        agent: AtomicUsize, // POST /v1/agents
        get: AtomicUsize,   // retrieve GETs (verification calls)
    }

    async fn spawn(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        format!("http://{addr}")
    }

    // A mock Anthropic that COUNTS create-POSTs and serves retrieve GETs (200 if `retrieve_ok`,
    // else 404 so a persisted id reads as "gone").
    fn mock(counts: Arc<Counts>, retrieve_ok: bool) -> Router {
        let (ce, ca, cg1, cg2) = (counts.clone(), counts.clone(), counts.clone(), counts.clone());
        Router::new()
            .route(
                "/v1/environments",
                post(move || {
                    let c = ce.clone();
                    async move { Json(json!({"id": format!("env-{}", c.env.fetch_add(1, Ordering::SeqCst))})) }
                }),
            )
            .route(
                "/v1/agents",
                post(move || {
                    let c = ca.clone();
                    async move { Json(json!({"id": format!("agent-{}", c.agent.fetch_add(1, Ordering::SeqCst))})) }
                }),
            )
            .route("/v1/sessions", post(|| async { Json(json!({"id": "sess1"})) }))
            .route(
                "/v1/environments/{id}",
                get(move |AxPath(id): AxPath<String>| {
                    let c = cg1.clone();
                    async move {
                        c.get.fetch_add(1, Ordering::SeqCst);
                        if retrieve_ok { (StatusCode::OK, Json(json!({"id": id}))).into_response() } else { StatusCode::NOT_FOUND.into_response() }
                    }
                }),
            )
            .route(
                "/v1/agents/{id}",
                get(move |AxPath(id): AxPath<String>| {
                    let c = cg2.clone();
                    async move {
                        c.get.fetch_add(1, Ordering::SeqCst);
                        if retrieve_ok { (StatusCode::OK, Json(json!({"id": id}))).into_response() } else { StatusCode::NOT_FOUND.into_response() }
                    }
                }),
            )
    }

    #[tokio::test]
    async fn persisted_ids_are_reused_across_instances() {
        std::env::set_var("ANTHROPIC_API_KEY", "test-key");
        let counts = Arc::new(Counts::default());
        let url = spawn(mock(counts.clone(), true)).await;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("managed_agents.json");

        // First "process": one env + one agent, reused within the process.
        let a = ManagedAgents::with_base_url(url.clone(), Some(path.clone()));
        a.create_session("haiku", None, &json!({})).await.unwrap();
        a.create_session("haiku", None, &json!({})).await.unwrap();
        assert_eq!(counts.env.load(Ordering::SeqCst), 1, "env created once");
        assert_eq!(counts.agent.load(Ordering::SeqCst), 1, "agent created once");
        assert!(path.exists(), "ids were persisted to disk");

        // Second "process" (fresh instance, SAME file): reuses — no new env/agent (the leak fix).
        let b = ManagedAgents::with_base_url(url.clone(), Some(path.clone()));
        b.create_session("haiku", None, &json!({})).await.unwrap();
        assert_eq!(counts.env.load(Ordering::SeqCst), 1, "env reused across restart, not recreated");
        assert_eq!(counts.agent.load(Ordering::SeqCst), 1, "agent reused across restart, not recreated");
        assert!(counts.get.load(Ordering::SeqCst) >= 2, "reused ids were verified-on-first-use");
    }

    #[tokio::test]
    async fn stale_persisted_ids_self_heal() {
        std::env::set_var("ANTHROPIC_API_KEY", "test-key");
        let counts = Arc::new(Counts::default());
        let url = spawn(mock(counts.clone(), false)).await; // retrieve -> 404: the ids are gone
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("managed_agents.json");
        std::fs::write(
            &path,
            json!({"env_id": "gone-env", "agents": {"claude-haiku-4-5": "gone-agent"}}).to_string(),
        )
        .unwrap();

        // The persisted ids 404 on verify -> woollama recreates instead of bricking the path.
        let a = ManagedAgents::with_base_url(url, Some(path));
        a.create_session("haiku", None, &json!({})).await.unwrap();
        assert_eq!(counts.env.load(Ordering::SeqCst), 1, "stale env recreated");
        assert_eq!(counts.agent.load(Ordering::SeqCst), 1, "stale agent recreated");
    }

    #[tokio::test]
    async fn no_persist_path_means_no_file() {
        std::env::set_var("ANTHROPIC_API_KEY", "test-key");
        let counts = Arc::new(Counts::default());
        let url = spawn(mock(counts.clone(), true)).await;
        // Default (opt-in OFF): create works, nothing persisted, no verify GETs.
        let a = ManagedAgents::with_base_url(url, None);
        a.create_session("haiku", None, &json!({})).await.unwrap();
        assert_eq!(counts.env.load(Ordering::SeqCst), 1);
        assert_eq!(counts.agent.load(Ordering::SeqCst), 1);
        assert_eq!(counts.get.load(Ordering::SeqCst), 0, "ephemeral mode never verifies/persists");
    }
}
