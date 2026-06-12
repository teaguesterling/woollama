//! Native Ollama `/api/chat` translation (issue #1) — pure functions ported from
//! Python `woollama.ollama_native`.
//!
//! Ollama's OpenAI-compat `/v1/chat/completions` IGNORES `options.num_ctx`; only the
//! native `/api/chat` honors it. So a passthrough `ollama/<model>` request that asks
//! for a context size (and carries no tools) is routed to `/api/chat`, translating the
//! request to native shape and the response back to OpenAI `chat.completion`.
//!
//! Slice 3 ports the NON-STREAM path. The NDJSON→SSE streaming translator is slice 3b.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};
use uuid::Uuid;

fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn chatcmpl_id() -> String {
    format!("chatcmpl-{}", Uuid::new_v4().simple())
}

fn finish(done_reason: Option<&str>) -> &'static str {
    if done_reason == Some("length") { "length" } else { "stop" }
}

/// True if this ollama request needs the native endpoint: `options.num_ctx` is set AND
/// it carries no tools (tool requests stay on `/v1`).
pub fn wants_native(body: &Value) -> bool {
    let has_ctx = body
        .get("options")
        .and_then(Value::as_object)
        .and_then(|o| o.get("num_ctx"))
        .is_some_and(|v| !v.is_null());
    let no_tools = match body.get("tools") {
        None | Some(Value::Null) => true,
        Some(Value::Array(a)) => a.is_empty(),
        Some(_) => false,
    };
    has_ctx && no_tools
}

/// Derive the native `/api/chat` URL from ollama's OpenAI-compat `base_url` (`<root>/v1`).
pub fn native_chat_url(base_url: &str) -> String {
    let root = base_url.trim_end_matches('/');
    let root = root.strip_suffix("/v1").unwrap_or(root);
    format!("{root}/api/chat")
}

/// OpenAI chat-completions body → ollama `/api/chat` body. `body["model"]` is already
/// the bare model name. Folds top-level sampling params into `options` (without
/// clobbering caller-set ones); `max_(completion_)tokens` → `num_predict`.
pub fn to_native_request(body: &Value) -> Value {
    let mut options: Map<String, Value> =
        body.get("options").and_then(Value::as_object).cloned().unwrap_or_default();
    for (oai, native) in [("temperature", "temperature"), ("top_p", "top_p"), ("seed", "seed"), ("stop", "stop")] {
        if let Some(v) = body.get(oai) {
            if !options.contains_key(native) {
                options.insert(native.to_string(), v.clone());
            }
        }
    }
    if !options.contains_key("num_predict") {
        for cap in ["max_completion_tokens", "max_tokens"] {
            if let Some(v) = body.get(cap) {
                if !v.is_null() {
                    options.insert("num_predict".to_string(), v.clone());
                    break;
                }
            }
        }
    }
    let mut req = json!({
        "model": body.get("model").cloned().unwrap_or_else(|| json!("")),
        "messages": body.get("messages").cloned().unwrap_or_else(|| json!([])),
        "stream": body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        "options": options,
    });
    if let Some(f) = body.get("format") {
        if !f.is_null() {
            req["format"] = f.clone();
        }
    }
    req
}

/// ollama `/api/chat` (stream:false) response → OpenAI `chat.completion`.
pub fn from_native_response(native: &Value, model: &str) -> Value {
    let content = native
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let prompt = native.get("prompt_eval_count").and_then(Value::as_i64).unwrap_or(0);
    let completion = native.get("eval_count").and_then(Value::as_i64).unwrap_or(0);
    json!({
        "id": chatcmpl_id(),
        "object": "chat.completion",
        "created": now_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish(native.get("done_reason").and_then(Value::as_str)),
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    })
}

/// Stateful NDJSON→SSE translator: ollama `/api/chat` stream frames → OpenAI
/// `chat.completion.chunk` SSE byte strings. The first content chunk carries
/// `role:assistant`; the `done:true` frame emits a finish chunk + `data: [DONE]`.
pub struct SseTranslator {
    cid: String,
    created: i64,
    model: String,
    role_sent: bool,
}

impl SseTranslator {
    pub fn new(model: &str) -> Self {
        SseTranslator { cid: chatcmpl_id(), created: now_secs(), model: model.to_string(), role_sent: false }
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> Vec<u8> {
        let payload = json!({
            "id": self.cid, "object": "chat.completion.chunk",
            "created": self.created, "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        });
        format!("data: {}\n\n", serde_json::to_string(&payload).unwrap()).into_bytes()
    }

    /// Translate one NDJSON frame line into zero or more SSE byte strings.
    pub fn translate(&mut self, line: &str) -> Vec<Vec<u8>> {
        let line = line.trim();
        if line.is_empty() {
            return vec![];
        }
        let Ok(frame) = serde_json::from_str::<Value>(line) else { return vec![] };
        if frame.get("done").and_then(Value::as_bool).unwrap_or(false) {
            return vec![
                self.chunk(json!({}), Some(finish(frame.get("done_reason").and_then(Value::as_str)))),
                b"data: [DONE]\n\n".to_vec(),
            ];
        }
        let content = frame.get("message").and_then(|m| m.get("content")).and_then(Value::as_str).unwrap_or("");
        if content.is_empty() {
            return vec![];
        }
        if !self.role_sent {
            self.role_sent = true;
            vec![self.chunk(json!({"role": "assistant", "content": content}), None)]
        } else {
            vec![self.chunk(json!({"content": content}), None)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wants_native_only_with_num_ctx_and_no_tools() {
        assert!(wants_native(&json!({"options": {"num_ctx": 16384}})));
        assert!(!wants_native(&json!({"options": {"temperature": 0}})));
        assert!(!wants_native(&json!({})));
        assert!(!wants_native(&json!({"options": {"num_ctx": 8192}, "tools": [{"type": "function"}]})));
        assert!(wants_native(&json!({"options": {"num_ctx": 8192}, "tools": []})));
    }

    #[test]
    fn native_chat_url_strips_v1() {
        assert_eq!(native_chat_url("http://localhost:11434/v1"), "http://localhost:11434/api/chat");
        assert_eq!(native_chat_url("http://host:11434/v1/"), "http://host:11434/api/chat");
    }

    #[test]
    fn to_native_folds_params_and_keeps_num_ctx() {
        let req = to_native_request(&json!({
            "model": "qwen3:14b", "messages": [{"role": "user", "content": "hi"}],
            "stream": true, "temperature": 0.5, "max_tokens": 256,
            "options": {"num_ctx": 16384}
        }));
        assert_eq!(req["model"], "qwen3:14b");
        assert_eq!(req["stream"], true);
        assert_eq!(req["options"]["num_ctx"], 16384);
        assert_eq!(req["options"]["temperature"], 0.5);
        assert_eq!(req["options"]["num_predict"], 256);
        assert_eq!(req["messages"], json!([{"role": "user", "content": "hi"}]));
    }

    #[test]
    fn to_native_does_not_clobber_caller_options() {
        let req = to_native_request(&json!({
            "model": "m", "temperature": 0.9, "options": {"num_ctx": 4096, "temperature": 0.1}
        }));
        assert_eq!(req["options"]["temperature"], 0.1);
    }

    #[test]
    fn from_native_maps_shape_and_usage() {
        let native = json!({
            "model": "qwen3.5:4b", "created_at": "2026-06-07T22:46:47Z",
            "message": {"role": "assistant", "content": "hello"},
            "done": true, "done_reason": "stop", "prompt_eval_count": 19, "eval_count": 20
        });
        let out = from_native_response(&native, "qwen3.5:4b");
        assert_eq!(out["object"], "chat.completion");
        assert!(out["created"].is_i64());
        assert_eq!(out["choices"][0]["message"], json!({"role": "assistant", "content": "hello"}));
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
        assert_eq!(out["usage"], json!({"prompt_tokens": 19, "completion_tokens": 20, "total_tokens": 39}));
    }

    #[test]
    fn from_native_maps_length_finish() {
        let out = from_native_response(
            &json!({"message": {"content": ""}, "done": true, "done_reason": "length"}),
            "m",
        );
        assert_eq!(out["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn sse_translator_role_then_deltas_then_done() {
        let mut t = SseTranslator::new("qwen3.5:4b");
        let lines = [
            r#"{"message":{"role":"assistant","content":"one"},"done":false}"#,
            r#"{"message":{"role":"assistant","content":" two"},"done":false}"#,
            r#"{"message":{"content":""},"done":true,"done_reason":"stop"}"#,
        ];
        let chunks: Vec<String> =
            lines.iter().flat_map(|l| t.translate(l)).map(|c| String::from_utf8(c).unwrap()).collect();
        let first: Value = serde_json::from_str(chunks[0].trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(first["choices"][0]["delta"], json!({"role": "assistant", "content": "one"}));
        let second: Value = serde_json::from_str(chunks[1].trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(second["choices"][0]["delta"], json!({"content": " two"}));
        let term: Value = serde_json::from_str(chunks[2].trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(term["choices"][0]["finish_reason"], "stop");
        assert_eq!(chunks[3], "data: [DONE]\n\n");
    }
}
