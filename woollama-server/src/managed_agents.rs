//! Anthropic Managed Agents — the `managed-agents` conversation backend (conv-6),
//! ported (woollama-side) from Python `woollama.managed_agents`.
//!
//! Anthropic owns the session/transcript/loop/container; woollama holds only the
//! session_id. Auth is `ANTHROPIC_API_KEY` (paid, distinct from the keyless
//! claude-resume/claude-code subscription path). One tool-less agent per model is
//! created lazily + cached (an agent is a reusable account object — never per session).
//!
//! ⚠️ WIRE-FORMAT CAVEAT: the Python version goes through the Anthropic SDK, so the exact
//! Managed Agents REST/streaming shapes are NOT in this repo. This client targets a
//! base-URL-overridable (`ANTHROPIC_BASE_URL`) endpoint with a SIMPLIFIED turn protocol
//! (`POST /v1/sessions/{id}/turns` → `{text, pending?}`; `GET …/events` → `{data:[…]}`),
//! which the hermetic test exercises via a mock. The real Anthropic API must be
//! reconciled here before the opt-in live `@needs_anthropic` test passes — THAT is the
//! gate for the actual wire format. The woollama-side ROUTING below is the tested value.

use std::collections::HashMap;

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
/// client-side `ask_user` tool (the interactive requires_action signal).
pub struct Turn {
    pub text: String,
    pub pending: Option<Pending>,
}

pub struct Pending {
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

    /// Lazily create + cache the shared environment and a per-model tool-less agent.
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
    pub async fn create_session(&self, model: &str, title: Option<&str>, metadata: &Value) -> Result<String, ManagedAgentsError> {
        let (agent_id, env_id) = self.ensure_agent(model).await?;
        let mut body = json!({"agent": agent_id, "environment_id": env_id});
        if let Some(t) = title {
            body["title"] = json!(t);
        }
        if metadata.is_object() && !metadata.as_object().unwrap().is_empty() {
            body["metadata"] = metadata.clone();
        }
        let s = self.post("/v1/sessions", body).await?;
        Ok(s["id"].as_str().unwrap_or_default().to_string())
    }

    fn parse_turn(v: &Value) -> Turn {
        let pending = v.get("pending").filter(|p| !p.is_null()).map(|p| Pending {
            id: p.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
            input: p.get("input").cloned().unwrap_or_else(|| json!({})),
        });
        Turn { text: v.get("text").and_then(Value::as_str).unwrap_or_default().to_string(), pending }
    }

    /// One conversational turn (a new user message).
    pub async fn run_turn(&self, session_id: &str, text: &str) -> Result<Turn, ManagedAgentsError> {
        let v = self.post(&format!("/v1/sessions/{session_id}/turns"), json!({"input": text})).await?;
        Ok(Self::parse_turn(&v))
    }

    /// Resume a paused session by returning the user's answer to the pending ask_user tool.
    pub async fn answer_turn(&self, session_id: &str, tool_use_id: &str, answer: &str) -> Result<Turn, ManagedAgentsError> {
        let v = self
            .post(&format!("/v1/sessions/{session_id}/turns"), json!({"tool_use_id": tool_use_id, "answer": answer}))
            .await?;
        Ok(Self::parse_turn(&v))
    }

    /// Retrieve the transcript (Anthropic owns the bytes; woollama reshapes to messages).
    pub async fn history(&self, session_id: &str) -> Result<Vec<Value>, ManagedAgentsError> {
        let r = self
            .request(reqwest::Method::GET, &format!("/v1/sessions/{session_id}/events"))?
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
