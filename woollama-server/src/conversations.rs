//! Conversation handle routing — the stateful surface's routing layer (slice 6a),
//! ported from Python `woollama.conversations`.
//!
//! The principle: **woollama routes conversation *handles*; the backends own the
//! *state*.** This is the durable handle table (`conv_id → backend + native_id`) plus
//! the claude-resume backend. woollama never stores the transcript.
//!
//! Slice 6a: the handle table (durable) + `claude-resume`. Store-backed providers
//! (ollama/cloud/recipe statefulness) → 6b; managed-agents → 7.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::responses;

fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// A routable handle. `native_id` is the backend's own id (a claude session_id), None
/// until the first turn creates the backing session. This is ROUTING state — not the
/// transcript — so persisting it doesn't violate "woollama owns no conversation state".
#[derive(Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub backend: String,
    pub model: String,
    #[serde(default)]
    pub native_id: Option<String>,
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub workdir: Option<String>,
    #[serde(default)]
    pub response_ids: Vec<String>,
    #[serde(default = "idle")]
    pub status: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub metadata: Value,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(default)]
    pub required_action: Option<Value>,
    #[serde(default)]
    pub pending_tool_use_id: Option<String>,
}

fn idle() -> String {
    "idle".to_string()
}

impl Conversation {
    /// The discovery object (OpenAI Conversation base + woollama routing extras).
    pub fn to_object(&self) -> Value {
        json!({
            "id": self.id,
            "object": "conversation",
            "created_at": self.created_at,
            "metadata": if self.metadata.is_null() { json!({}) } else { self.metadata.clone() },
            "backend": self.backend,
            "model": self.model,
            "status": self.status,
            "title": self.title,
            "key": self.key,
            "updated_at": self.updated_at,
        })
    }
}

/// The durable handle table: `conv_id → Conversation`, `resp_id → conv_id`, and
/// `key → conv_id`. With a `path` it's atomically rewritten on every mutation so
/// conv_ids survive a restart; without one it's purely in-memory.
pub struct ConversationStore {
    convs: HashMap<String, Conversation>,
    resp_to_conv: HashMap<String, String>,
    alias_to_conv: HashMap<String, String>,
    path: Option<PathBuf>,
}

impl ConversationStore {
    pub fn new(path: Option<PathBuf>) -> Self {
        let mut s = ConversationStore {
            convs: HashMap::new(),
            resp_to_conv: HashMap::new(),
            alias_to_conv: HashMap::new(),
            path,
        };
        s.load();
        s
    }

    fn save(&self) {
        let Some(path) = &self.path else { return };
        let convs: Vec<&Conversation> = self.convs.values().collect();
        let data = json!({"convs": convs, "resp_to_conv": self.resp_to_conv});
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, data.to_string()).is_ok() {
            let _ = std::fs::rename(&tmp, path); // atomic swap
        }
    }

    fn load(&mut self) {
        let Some(path) = &self.path else { return };
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let Ok(data) = serde_json::from_str::<Value>(&text) else { return };
        if let Some(arr) = data.get("convs").and_then(Value::as_array) {
            for d in arr {
                let mut d = d.clone();
                // A persisted 'busy' is stale after a restart — reset so it's usable.
                if d.get("status").and_then(Value::as_str) == Some("busy") {
                    d["status"] = json!("idle");
                }
                if let Ok(conv) = serde_json::from_value::<Conversation>(d) {
                    if let Some(k) = &conv.key {
                        self.alias_to_conv.insert(k.clone(), conv.id.clone());
                    }
                    self.convs.insert(conv.id.clone(), conv);
                }
            }
        }
        if let Some(map) = data.get("resp_to_conv").and_then(Value::as_object) {
            for (k, v) in map {
                if let Some(s) = v.as_str() {
                    self.resp_to_conv.insert(k.clone(), s.to_string());
                }
            }
        }
    }

    pub fn create(
        &mut self,
        backend: &str,
        model: &str,
        metadata: Value,
        title: Option<String>,
        key: Option<String>,
    ) -> Conversation {
        let now = now_secs();
        let conv = Conversation {
            id: responses::new_id("conv"),
            backend: backend.to_string(),
            model: model.to_string(),
            native_id: None,
            key: key.clone(),
            workdir: None,
            response_ids: Vec::new(),
            status: "idle".to_string(),
            title,
            metadata,
            created_at: now,
            updated_at: now,
            required_action: None,
            pending_tool_use_id: None,
        };
        if let Some(k) = key {
            self.alias_to_conv.insert(k, conv.id.clone());
        }
        self.convs.insert(conv.id.clone(), conv.clone());
        self.save();
        conv
    }

    pub fn by_alias(&self, key: &str) -> Option<Conversation> {
        self.alias_to_conv.get(key).and_then(|cid| self.convs.get(cid)).cloned()
    }

    pub fn get_or_create_by_alias(&mut self, key: &str, backend: &str, model: &str) -> Conversation {
        if let Some(existing) = self.by_alias(key) {
            return existing;
        }
        self.create(backend, model, json!({}), None, Some(key.to_string()))
    }

    pub fn get(&self, conv_id: &str) -> Option<Conversation> {
        self.convs.get(conv_id).cloned()
    }

    pub fn list(&self) -> Vec<Conversation> {
        self.convs.values().cloned().collect()
    }

    pub fn by_response(&self, response_id: &str) -> Option<Conversation> {
        self.resp_to_conv.get(response_id).and_then(|cid| self.convs.get(cid)).cloned()
    }

    pub fn record_response(&mut self, conv_id: &str, response_id: &str) {
        if let Some(conv) = self.convs.get_mut(conv_id) {
            conv.response_ids.push(response_id.to_string());
            conv.updated_at = now_secs();
        }
        self.resp_to_conv.insert(response_id.to_string(), conv_id.to_string());
        self.save();
    }

    /// Update the backend's native id + workdir after a turn (the persistence point that
    /// captures the claude session_id, so a restart can still resume it).
    pub fn set_native(&mut self, conv_id: &str, native_id: Option<String>, workdir: Option<String>) {
        if let Some(conv) = self.convs.get_mut(conv_id) {
            conv.native_id = native_id;
            conv.workdir = workdir;
            conv.updated_at = now_secs();
        }
        self.save();
    }

    pub fn remove(&mut self, conv_id: &str) -> Option<Conversation> {
        let conv = self.convs.remove(conv_id);
        if let Some(c) = &conv {
            for rid in &c.response_ids {
                self.resp_to_conv.remove(rid);
            }
            if let Some(k) = &c.key {
                self.alias_to_conv.remove(k);
            }
            self.save();
        }
        conv
    }
}

/// The handle table behind a mutex, plus per-conversation locks (one writer per
/// conversation — the backing session is single-threaded).
pub struct Conversations {
    pub table: tokio::sync::Mutex<ConversationStore>,
    locks: tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl Conversations {
    pub fn new(path: Option<PathBuf>) -> Self {
        Conversations {
            table: tokio::sync::Mutex::new(ConversationStore::new(path)),
            locks: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    /// The per-conversation write lock (created on first use).
    pub async fn conv_lock(&self, conv_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.lock().await;
        locks.entry(conv_id.to_string()).or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))).clone()
    }
}

/// Which state-owning backend (if any) backs conversations for this model:
/// `claude-code/<model>` → `claude-resume`; every other model → `store-backed` IFF a
/// conversation-store provider is wired (`has_store`), else stateless. (`claude-agent`
/// → managed-agents is slice 7.)
pub fn backend_for_model(model: &str, has_store: bool) -> Option<&'static str> {
    match model.split('/').next().unwrap_or("") {
        "claude-code" => Some("claude-resume"),
        _ => has_store.then_some("store-backed"),
    }
}

// --- external conversation stores (issue #2): woollama is a CLIENT; the store owns
//     the transcript bytes ------------------------------------------------------

use std::time::Duration;

use woollama_engine::EngineError;

use crate::mcp_registry::McpRegistry;

/// A pluggable external owner of conversation transcripts. woollama assembles prior
/// history + the new turn and runs STATELESS inference; the store owns the bytes.
#[async_trait::async_trait]
pub trait StoreProvider: Send + Sync {
    async fn create(&self) -> Result<String, EngineError>;
    async fn get(&self, thread_id: &str) -> Result<Vec<Value>, EngineError>;
    async fn append(&self, thread_id: &str, messages: &Value) -> Result<(), EngineError>;
    async fn delete(&self, thread_id: &str) -> Result<(), EngineError>;
}

/// A `StoreProvider` over a REST conversation-store endpoint (examples/rest-convstore).
/// The provider mints the thread id (a uuid) and PUTs it, so create is idempotent.
pub struct HttpStoreProvider {
    base: String,
}

impl HttpStoreProvider {
    pub fn new(url: &str) -> Self {
        HttpStoreProvider { base: url.trim_end_matches('/').to_string() }
    }
    async fn req(&self, method: &str, path: &str, body: Option<Value>) -> Result<Option<Value>, EngineError> {
        let fail = |e: String| EngineError::new(
            format!("conversation store (http {}) failed on {method} {path}: {e}", self.base),
            "upstream_error",
            502,
        );
        let client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build().map_err(|e| fail(e.to_string()))?;
        let m = reqwest::Method::from_bytes(method.as_bytes()).unwrap_or(reqwest::Method::GET);
        let mut rb = client.request(m, format!("{}{path}", self.base));
        if let Some(b) = body {
            rb = rb.json(&b);
        }
        let r = rb.send().await.map_err(|e| fail(e.to_string()))?;
        if !r.status().is_success() {
            return Err(fail(format!("status {}", r.status())));
        }
        let bytes = r.bytes().await.map_err(|e| fail(e.to_string()))?;
        if bytes.is_empty() {
            return Ok(None);
        }
        Ok(serde_json::from_slice(&bytes).ok())
    }
}

#[async_trait::async_trait]
impl StoreProvider for HttpStoreProvider {
    async fn create(&self) -> Result<String, EngineError> {
        let id = uuid::Uuid::new_v4().simple().to_string();
        self.req("PUT", &format!("/threads/{id}"), None).await?;
        Ok(id)
    }
    async fn get(&self, thread_id: &str) -> Result<Vec<Value>, EngineError> {
        Ok(self
            .req("GET", &format!("/threads/{thread_id}"), None)
            .await?
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default())
    }
    async fn append(&self, thread_id: &str, messages: &Value) -> Result<(), EngineError> {
        self.req("PATCH", &format!("/threads/{thread_id}"), Some(messages.clone())).await?;
        Ok(())
    }
    async fn delete(&self, thread_id: &str) -> Result<(), EngineError> {
        self.req("DELETE", &format!("/threads/{thread_id}"), None).await?;
        Ok(())
    }
}

/// A `StoreProvider` over an MCP conversation-store server (examples/mcp-convstore): each
/// op is one MCP tool call whose JSON text block is the result.
pub struct McpStoreProvider {
    reg: Arc<McpRegistry>,
    server: String,
}

impl McpStoreProvider {
    pub fn new(reg: Arc<McpRegistry>, server: String) -> Self {
        McpStoreProvider { reg, server }
    }
    async fn call(&self, tool: &str, args: Value) -> Result<Value, EngineError> {
        let fail = |e: String| EngineError::new(
            format!("conversation store '{}' failed on '{tool}': {e}", self.server),
            "upstream_error",
            502,
        );
        let res = self.reg.call_server(&self.server, tool, &args).await.map_err(fail)?;
        let text: String = res.content.iter().filter_map(|c| c.as_text().map(|t| t.text.clone())).collect();
        serde_json::from_str(&text).map_err(|e| fail(format!("bad json: {e}")))
    }
}

#[async_trait::async_trait]
impl StoreProvider for McpStoreProvider {
    async fn create(&self) -> Result<String, EngineError> {
        let v = self.call("create_thread", json!({})).await?;
        v.as_str()
            .map(String::from)
            .ok_or_else(|| EngineError::new("create_thread did not return a thread id", "upstream_error", 502))
    }
    async fn get(&self, thread_id: &str) -> Result<Vec<Value>, EngineError> {
        Ok(self.call("get_thread", json!({"thread_id": thread_id})).await?.as_array().cloned().unwrap_or_default())
    }
    async fn append(&self, thread_id: &str, messages: &Value) -> Result<(), EngineError> {
        self.call("append_turn", json!({"thread_id": thread_id, "messages": messages})).await?;
        Ok(())
    }
    async fn delete(&self, thread_id: &str) -> Result<(), EngineError> {
        self.call("delete_thread", json!({"thread_id": thread_id})).await?;
        Ok(())
    }
}
