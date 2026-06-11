//! A fake `claude` CLI for hermetic claude-code tests. Ignores its args/stdin and prints
//! one `claude -p --output-format json` result event so `claude_code::run_completion`
//! parses a deterministic answer. Real-CLI behavior (auth, the lockdown actually holding,
//! delegation tool calls) is covered by the opt-in live tests in a plain terminal.

fn main() {
    println!(r#"[{{"type":"result","result":"fake-answer","is_error":false,"session_id":"s1"}}]"#);
}
