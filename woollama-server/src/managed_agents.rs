//! Anthropic Managed Agents — the `managed-agents` conversation backend (conv-6),
//! ported (woollama-side) from Python `woollama.managed_agents`.
//!
//! Anthropic owns the session/transcript/loop/container; woollama holds only the
//! session_id. Auth is `ANTHROPIC_API_KEY` (paid, distinct from the keyless
//! claude-resume/claude-code subscription path). One tool-less agent per model is
//! created lazily + cached (an agent is a reusable account object — never per session).
//!
//! WIRE FORMAT (reconciled 2026-06-15 against the official docs — Managed Agents is
//! **event-driven + SSE-streamed**, NOT a synchronous turn endpoint):
//!
//! - create session: `POST /v1/sessions {agent, environment_id}` → `{id}`.
//! - send a turn: `POST /v1/sessions/{id}/events?beta=true` with `{events:[{type:"user.message", content:[{type:"text",text}]}]}`.
//! - read the answer: `GET /v1/sessions/{id}/events?beta=true` with `accept: text/event-stream` → SSE; assistant text arrives as `agent.message` events; the turn ends at `session.status_idle` (`stop_reason.type` = `end_turn`, or `requires_action` when the agent paused on a custom tool, with `stop_reason.event_ids` listing the blocking `agent.custom_tool_use` event ids).
//! - resume (ask_user): `POST …/events?beta=true` with `{events:[{type:"user.custom_tool_result", custom_tool_use_id, content:[{type:"text",text}]}]}`.
//! - history: `GET /v1/sessions/{id}/events?beta=true` (list).
//!
//! Docs: platform.claude.com/docs/en/managed-agents/{overview,sessions,events-and-streaming}.
//!
//! ⚠️ CONFIDENCE: this matches the docs but is **NOT live-verified** — there's no paid key in
//! the repo. The opt-in `@needs_anthropic` integration test is the real gate; known unknowns
//! it must confirm: the GET-events stream's replay semantics (do we over-collect prior turns'
//! text?), the exact `agent.custom_tool_use` id field, the custom-tool agent-config shape, and
//! the list-events (history) response envelope. The woollama-side ROUTING is the tested value.

use std::collections::HashMap;

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
/// agents (created once, never per session — the documented anti-pattern).
pub struct ManagedAgents {
    base_url: String,
    state: Mutex<Setup>,
}

#[derive(Default)]
struct Setup {
    env_id: Option<String>,
    agents: HashMap<String, String>, // full model id → agent_id
}

impl ManagedAgents {
    pub fn new() -> Self {
        let base_url = std::env::var("ANTHROPIC_BASE_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());
        ManagedAgents { base_url: base_url.trim_end_matches('/').to_string(), state: Mutex::new(Setup::default()) }
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

    /// Lazily create + cache the shared environment and a per-model agent.
    async fn ensure_agent(&self, model: &str) -> Result<(String, String), ManagedAgentsError> {
        let full = resolve_model(model);
        let mut s = self.state.lock().await;
        if s.env_id.is_none() {
            let env = self.post("/v1/environments", json!({
                "name": "woollama-agents",
                "config": {"type": "cloud", "networking": {"type": "unrestricted"}}
            })).await?;
            s.env_id = Some(env["id"].as_str().unwrap_or_default().to_string());
        }
        if !s.agents.contains_key(&full) {
            let agent = self.post("/v1/agents", json!({
                "name": format!("woollama:{full}"),
                "model": full,
                "tools": [ask_user_tool()],
            })).await?;
            s.agents.insert(full.clone(), agent["id"].as_str().unwrap_or_default().to_string());
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
            .request(reqwest::Method::GET, &format!("/v1/sessions/{session_id}/events?beta=true"))?
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
        self.request(reqwest::Method::DELETE, &format!("/v1/sessions/{session_id}"))?
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
