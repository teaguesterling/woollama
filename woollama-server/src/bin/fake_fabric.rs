//! A fake `fabric` CLI for hermetic vision-path tests. Real fabric (ground-truthed against
//! llama3.2-vision) reads userInput from stdin and prints a plain-text answer to stdout; this
//! mirrors that contract and echoes back the `--pattern` / `--attachment` it was given plus the
//! stdin, so a test can assert the argv assembly and stdin both reached the subprocess. The real
//! CLI + a real vision model are exercised manually (see docs/patterns.md); this keeps CI hermetic.

use std::io::Read;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    // We only emit the `=`-form (`--pattern=foo`) from `build_fabric_argv`, so match that.
    args.iter().find_map(|a| a.strip_prefix(&format!("{flag}=")).map(String::from))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut stdin = String::new();
    let _ = std::io::stdin().read_to_string(&mut stdin);
    let pattern = arg_value(&args, "--pattern").unwrap_or_default();
    let attachment = arg_value(&args, "--attachment").unwrap_or_default();
    // Plain text on stdout, exactly like real fabric.
    print!("VISION pattern={pattern} att={attachment} input={}", stdin.trim());
}
