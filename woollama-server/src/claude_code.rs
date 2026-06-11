//! Claude Code as an inference backend — tool-less completions AND delegation,
//! ported from Python `woollama.claude_code`.
//!
//! A recipe whose inferencer is `claude-code/<model>` routes to the local `claude` CLI
//! in headless print mode, using the user's EXISTING Claude auth (no ANTHROPIC_API_KEY).
//! Two modes: tool-less `run_completion` (empty `tools`), and delegation `run_delegated`
//! (non-empty `tools` — Claude owns the agentic loop and calls the recipe's allow-listed
//! MCP tools itself; woollama returns only the final answer).
//!
//! Safety (unchanged from the Python lockdown): the built-in tool set is an allow-list of
//! NONE (`--tools ""`), with `_DENY_TOOLS` as defense-in-depth, `--permission-mode
//! dontAsk`, `--setting-sources project` (don't inherit host ~/.claude), a neutral temp
//! cwd, and an ALLOW-LISTED child env (no provider keys / secrets / parent-harness vars).
//! Delegation additionally writes a per-recipe `--mcp-config` with ONLY the referenced
//! servers and `--allowedTools` listing ONLY the recipe's tools.

use std::collections::HashMap;
use std::time::Duration;

use serde_json::{json, Value};

const DENY_TOOLS: &str = "Bash,Read,Write,Edit,NotebookEdit,WebFetch,WebSearch,Glob,Grep,Task,LSP";

/// Operational vars only — no provider keys / secrets / parent-harness vars reach the
/// `claude` child or the MCP servers it spawns. (ANTHROPIC_API_KEY and CLAUDE_CODE*/
/// CLAUDECODE are deliberately absent.)
const CHILD_ENV_ALLOW: &[&str] = &[
    "HOME", "PATH", "USER", "LOGNAME", "SHELL", "TERM", "TZ", "TMPDIR", "LANG", "LANGUAGE",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
];

fn claude_bin() -> String {
    std::env::var("WOOLLAMA_CLAUDE_BIN").unwrap_or_else(|_| "claude".to_string())
}

#[derive(Debug)]
pub struct ClaudeCodeError(pub String);

impl std::fmt::Display for ClaudeCodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Allow-listed env for the child (+ `ENABLE_TOOL_SEARCH=false` so the recipe's MCP
/// tools load upfront, since `--tools ""` disables the deferred-tool search).
pub fn child_env() -> HashMap<String, String> {
    let mut env: HashMap<String, String> = std::env::vars()
        .filter(|(k, _)| CHILD_ENV_ALLOW.contains(&k.as_str()) || k.starts_with("LC_"))
        .collect();
    env.insert("ENABLE_TOOL_SEARCH".to_string(), "false".to_string());
    env
}

/// `<server>.<tool>` → `mcp__<server>__<tool>`. Rejects commas/whitespace (an
/// `--allowedTools` entry is comma-joined; a crafted name must not inject a second entry).
pub fn mcp_tool_name(namespaced: &str) -> Result<String, String> {
    if namespaced.contains(',') || namespaced.chars().any(char::is_whitespace) {
        return Err(format!(
            "invalid tool name in recipe allow-list: {namespaced:?} (commas/whitespace not allowed)"
        ));
    }
    let (server, tool) = namespaced.split_once('.').unwrap_or((namespaced, ""));
    Ok(format!("mcp__{server}__{tool}"))
}

/// Flatten OpenAI messages into one prompt for `claude -p` (system messages dropped —
/// the recipe's system prompt is passed via `--system-prompt`).
fn render_prompt(user_msgs: &Value) -> String {
    let msgs: Vec<&Value> = user_msgs
        .as_array()
        .map(|a| a.iter().filter(|m| m.get("role").and_then(Value::as_str) != Some("system")).collect())
        .unwrap_or_default();
    if msgs.len() == 1 {
        return msgs[0].get("content").and_then(Value::as_str).unwrap_or("").to_string();
    }
    msgs.iter()
        .map(|m| {
            format!(
                "{}: {}",
                m.get("role").and_then(Value::as_str).unwrap_or("user"),
                m.get("content").and_then(Value::as_str).unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_args(prompt: &str, system: &str, model: &str) -> Vec<String> {
    let mut args = vec![
        claude_bin(), "-p".into(), prompt.into(),
        "--output-format".into(), "json".into(),
        "--max-turns".into(), "1".into(),
        "--strict-mcp-config".into(),
        "--setting-sources".into(), "project".into(),
        "--permission-mode".into(), "dontAsk".into(),
        "--tools".into(), "".into(),
        "--disallowedTools".into(), DENY_TOOLS.into(),
    ];
    if !system.is_empty() {
        args.push("--system-prompt".into());
        args.push(system.into());
    }
    if !model.is_empty() {
        args.push("--model".into());
        args.push(model.into());
    }
    args
}

fn build_delegate_args(
    prompt: &str,
    system: &str,
    model: &str,
    mcp_config_path: &str,
    allowed: &[String],
    max_turns: u32,
) -> Vec<String> {
    let mut args = vec![
        claude_bin(), "-p".into(), prompt.into(),
        "--output-format".into(), "json".into(),
        "--max-turns".into(), max_turns.to_string(),
        "--mcp-config".into(), mcp_config_path.into(),
        "--strict-mcp-config".into(),
        "--setting-sources".into(), "project".into(),
        "--permission-mode".into(), "dontAsk".into(),
        "--tools".into(), "".into(),
        "--disallowedTools".into(), DENY_TOOLS.into(),
        "--allowedTools".into(), allowed.join(","),
    ];
    if !system.is_empty() {
        args.push("--system-prompt".into());
        args.push(system.into());
    }
    if !model.is_empty() {
        args.push("--model".into());
        args.push(model.into());
    }
    args
}

/// Parse `claude -p --output-format json` (a JSON array of events) → (text, is_error,
/// session_id) from the terminal `result` event.
fn extract(stdout: &str) -> Result<(String, bool, Option<String>), ClaudeCodeError> {
    let data: Value = serde_json::from_str(stdout)
        .map_err(|e| ClaudeCodeError(format!("could not parse claude output: {e}")))?;
    let events = match data {
        Value::Array(a) => a,
        other => vec![other],
    };
    for ev in events.iter().rev() {
        if ev.get("type").and_then(Value::as_str) == Some("result") {
            return Ok((
                ev.get("result").and_then(Value::as_str).unwrap_or("").to_string(),
                ev.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                ev.get("session_id").and_then(Value::as_str).map(String::from),
            ));
        }
    }
    Err(ClaudeCodeError("no 'result' event in claude output".to_string()))
}

async fn invoke(
    args: &[String],
    env: &HashMap<String, String>,
    cwd: &str,
    timeout: f64,
) -> Result<(i32, Vec<u8>, Vec<u8>), ClaudeCodeError> {
    let mut cmd = tokio::process::Command::new(&args[0]);
    cmd.args(&args[1..])
        .current_dir(cwd)
        .env_clear() // the allow-listed env REPLACES the inherited one
        .envs(env)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let child = cmd
        .spawn()
        .map_err(|e| ClaudeCodeError(format!("`{}` not found on PATH: {e}", args[0])))?;
    let out = tokio::time::timeout(Duration::from_secs_f64(timeout), child.wait_with_output())
        .await
        .map_err(|_| ClaudeCodeError(format!("claude timed out after {timeout}s")))?
        .map_err(|e| ClaudeCodeError(format!("claude spawn error: {e}")))?;
    Ok((out.status.code().unwrap_or(-1), out.stdout, out.stderr))
}

async fn invoke_and_parse(
    args: &[String],
    env: &HashMap<String, String>,
    cwd: &str,
    timeout: f64,
) -> Result<(String, Option<String>), ClaudeCodeError> {
    let (rc, out, err) = invoke(args, env, cwd, timeout).await?;
    if rc != 0 {
        let tail: String = String::from_utf8_lossy(&err).chars().take(300).collect();
        return Err(ClaudeCodeError(format!("claude exited {rc}: {tail}")));
    }
    let (text, is_error, sid) = extract(&String::from_utf8_lossy(&out))?;
    if is_error {
        let tail: String = text.chars().take(300).collect();
        return Err(ClaudeCodeError(format!("claude returned an error result: {tail}")));
    }
    Ok((text, sid))
}

fn as_openai(text: &str) -> Value {
    json!({
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": text},
        }],
    })
}

/// One-shot, tool-less completion → an OpenAI chat-completions dict.
pub async fn run_completion(system: &str, user_msgs: &Value, model: &str) -> Result<Value, ClaudeCodeError> {
    let args = build_args(&render_prompt(user_msgs), system, model);
    let env = child_env();
    let tmp = tempfile::tempdir().map_err(|e| ClaudeCodeError(e.to_string()))?;
    let cwd = tmp.path().to_string_lossy().to_string();
    let (text, _) = invoke_and_parse(&args, &env, &cwd, 180.0).await?;
    Ok(as_openai(&text))
}

/// Delegated executor turn: hand Claude the recipe's allow-listed MCP tools (config
/// containment + `--allowedTools` + the built-in lockdown) and let it run the loop.
/// `mcp_servers` is `{server: {command, args}}` for the referenced servers only.
pub async fn run_delegated(
    system: &str,
    user_msgs: &Value,
    model: &str,
    allowed_tools: &[String],
    mcp_servers: &HashMap<String, Value>,
    max_turns: u32,
) -> Result<Value, ClaudeCodeError> {
    let allowed: Result<Vec<String>, String> = allowed_tools.iter().map(|t| mcp_tool_name(t)).collect();
    let allowed = allowed.map_err(ClaudeCodeError)?;
    let env = child_env();
    let tmp = tempfile::tempdir().map_err(|e| ClaudeCodeError(e.to_string()))?;
    let cwd = tmp.path().to_string_lossy().to_string();
    let cfg_path = tmp.path().join("delegate-mcp.json");
    std::fs::write(&cfg_path, json!({"mcpServers": mcp_servers}).to_string())
        .map_err(|e| ClaudeCodeError(e.to_string()))?;
    let args = build_delegate_args(
        &render_prompt(user_msgs),
        system,
        model,
        &cfg_path.to_string_lossy(),
        &allowed,
        max_turns,
    );
    let (text, _) = invoke_and_parse(&args, &env, &cwd, 300.0).await?;
    Ok(as_openai(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_tool_name_maps_and_rejects_injection() {
        assert_eq!(mcp_tool_name("hello.count_to").unwrap(), "mcp__hello__count_to");
        assert!(mcp_tool_name("count_to,mcp__hello__hello").is_err()); // comma injection
        assert!(mcp_tool_name("hello.count to").is_err()); // whitespace
    }

    #[test]
    fn build_args_carries_the_lockdown() {
        let a = build_args("hi", "be brief", "haiku");
        let s = a.join(" ");
        assert!(a.windows(2).any(|w| w == ["--tools", ""]), "allow-list of none");
        assert!(s.contains("--strict-mcp-config"));
        assert!(s.contains("--permission-mode dontAsk"));
        assert!(s.contains("--setting-sources project"));
        assert!(s.contains("--disallowedTools Bash,Read"));
        assert!(s.contains("--max-turns 1"));
        assert!(s.contains("--system-prompt be brief"));
        assert!(s.contains("--model haiku"));
    }

    #[test]
    fn delegate_args_allowlist_is_exactly_the_recipe_tools() {
        // The adversarial boundary: tools=[hello.count_to] grants ONLY that, never a
        // sibling/other-server tool — woollama can't widen Claude's grant.
        let allowed: Vec<String> = ["hello.count_to"].iter().map(|t| mcp_tool_name(t).unwrap()).collect();
        let a = build_delegate_args("hi", "sys", "haiku", "/tmp/cfg.json", &allowed, 8);
        let i = a.iter().position(|x| x == "--allowedTools").unwrap();
        assert_eq!(a[i + 1], "mcp__hello__count_to");
        assert!(!a[i + 1].contains("textops"));
        assert!(!a[i + 1].contains("mcp__hello__hello"));
        assert!(a.windows(2).any(|w| w == ["--mcp-config", "/tmp/cfg.json"]));
        assert!(a.windows(2).any(|w| w == ["--max-turns", "8"])); // not 1
        assert!(a.windows(2).any(|w| w == ["--tools", ""])); // lockdown kept
    }

    #[test]
    fn child_env_is_an_allow_list_with_tool_search_off() {
        let env = child_env();
        assert_eq!(env.get("ENABLE_TOOL_SEARCH").map(String::as_str), Some("false"));
        for k in env.keys() {
            assert!(
                CHILD_ENV_ALLOW.contains(&k.as_str()) || k.starts_with("LC_") || k == "ENABLE_TOOL_SEARCH",
                "leaked non-allow-listed env var: {k}"
            );
        }
    }

    #[test]
    fn extract_reads_the_result_event() {
        let (text, err, sid) = extract(
            r#"[{"type":"assistant"},{"type":"result","result":"pong","is_error":false,"session_id":"s1"}]"#,
        )
        .unwrap();
        assert_eq!(text, "pong");
        assert!(!err);
        assert_eq!(sid.as_deref(), Some("s1"));
        assert!(extract(r#"[{"type":"assistant"}]"#).is_err());
    }
}
