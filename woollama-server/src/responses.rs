//! OpenAI *Responses* wire-shape helpers — ported from Python `woollama.responses`
//! (the pure shaping layer). Slice 3 covers the STATELESS, non-stream subset:
//! `parse_input` + `build_response`. Stateful handle routing is slices 6/7; Responses
//! streaming + transcript items are slice 3b.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use uuid::Uuid;

fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// `resp_<hex>` / `msg_<hex>` — opaque per-turn handles.
pub fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4().simple())
}

/// Normalize the Responses `input` (a bare string OR a list of message items) into
/// OpenAI chat messages. A string is one user turn; a list maps each {role, content},
/// flattening content-part arrays ({type: input_text|output_text|text, text}) to text.
pub fn parse_input(value: &Value) -> Result<Vec<Value>, String> {
    match value {
        Value::String(s) => Ok(vec![json!({"role": "user", "content": s})]),
        Value::Array(items) => {
            let mut msgs = Vec::new();
            for item in items {
                let Some(obj) = item.as_object() else { continue };
                let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
                let content = match obj.get("content") {
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|p| {
                            let p = p.as_object()?;
                            let ty = p.get("type").and_then(Value::as_str)?;
                            if matches!(ty, "input_text" | "output_text" | "text") {
                                Some(p.get("text").and_then(Value::as_str).unwrap_or(""))
                            } else {
                                None
                            }
                        })
                        .collect::<String>(),
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                    None => String::new(),
                };
                msgs.push(json!({"role": role, "content": content}));
            }
            Ok(msgs)
        }
        _ => Err("`input` must be a string or a list of message items".to_string()),
    }
}

/// Assemble an OpenAI-Responses-shaped dict (the stateless/completed variant — the
/// `openai` SDK validates it; `.output_text` is its computed join of the parts).
pub fn build_response(resp_id: &str, model: &str, text: &str) -> Value {
    json!({
        "id": resp_id,
        "object": "response",
        "created_at": now_secs(),
        "model": model,
        "status": "completed",
        "conversation": Value::Null,
        "output": [{
            "type": "message",
            "id": new_id("msg"),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "parallel_tool_calls": false,
        "tool_choice": "auto",
        "tools": [],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_input_string_is_one_user_turn() {
        assert_eq!(parse_input(&json!("hi")).unwrap(), vec![json!({"role": "user", "content": "hi"})]);
    }

    #[test]
    fn parse_input_flattens_content_parts() {
        let msgs = parse_input(&json!([
            {"role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": "plain"}
        ]))
        .unwrap();
        assert_eq!(msgs[0], json!({"role": "user", "content": "ab"}));
        assert_eq!(msgs[1], json!({"role": "assistant", "content": "plain"}));
    }

    #[test]
    fn parse_input_rejects_scalar() {
        assert!(parse_input(&json!(42)).is_err());
    }

    #[test]
    fn build_response_has_responses_shape() {
        let r = build_response("resp_x", "ollama/qwen", "pong");
        assert_eq!(r["object"], "response");
        assert_eq!(r["status"], "completed");
        assert_eq!(r["conversation"], Value::Null);
        assert_eq!(r["output"][0]["content"][0]["text"], "pong");
        assert_eq!(r["output"][0]["content"][0]["type"], "output_text");
    }
}
