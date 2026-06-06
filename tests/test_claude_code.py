"""Unit tests for the Claude Code inference backend (claude_code.py).

These are hermetic: `claude_code._invoke` (the subprocess seam) is patched so no
real `claude` is spawned. They assert (a) the command woollama builds is locked
DOWN to tool-less — `--strict-mcp-config`, `--permission-mode dontAsk`, and
`--disallowedTools` covering Bash et al. — (b) ANTHROPIC_API_KEY is stripped
from the child env (so subscription auth is used), and (c) the JSON-array output
is parsed to OpenAI shape, with failures surfaced as ClaudeCodeError.

The RUNTIME safety claim (that the locked-down command actually refuses a tool)
can't be asserted here — that's the opt-in @needs_claude_code live test in
test_integration.py.
"""
from __future__ import annotations

import json

import pytest

from woollama import claude_code

# asyncio_mode=auto runs async tests automatically; no module-wide mark needed
# (and a module-wide asyncio mark would wrongly warn on the sync render tests).


def _result_array(text: str = "pong", is_error: bool = False) -> bytes:
    """A minimal stand-in for `claude -p --output-format json` (an ARRAY of
    events ending in a `result` event), matching the real v2.1.160 shape."""
    return json.dumps([
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
        {"type": "result", "subtype": "success", "is_error": is_error,
         "result": text, "total_cost_usd": 0.01},
    ]).encode()


def _patch_invoke(monkeypatch, *, rc=0, out=b"", err=b""):
    """Patch _invoke to capture (args, env, cwd) and return canned output."""
    captured: dict = {}

    async def fake_invoke(args, env, cwd, timeout):
        captured["args"] = args
        captured["env"] = env
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        return rc, out, err

    monkeypatch.setattr(claude_code, "_invoke", fake_invoke)
    return captured


# ---------------------------------------------------------------------------
# Command construction — the tool-less lockdown
# ---------------------------------------------------------------------------

async def test_builds_tool_less_locked_down_command(monkeypatch):
    captured = _patch_invoke(monkeypatch, out=_result_array("pong"))

    resp = await claude_code.run_completion(
        "You are a counter.", [{"role": "user", "content": "hi"}], "haiku")

    # OpenAI-shaped result, final text extracted from the `result` event.
    assert resp["choices"][0]["message"]["content"] == "pong"
    assert resp["choices"][0]["message"]["role"] == "assistant"

    args = captured["args"]
    assert args[:3] == [claude_code.CLAUDE_BIN, "-p", "hi"]
    # Tool-less lockdown is present.
    assert "--strict-mcp-config" in args
    assert args[args.index("--permission-mode") + 1] == "dontAsk"
    # PRIMARY lockdown: an allow-list of built-in tools set to NONE.
    assert args[args.index("--tools") + 1] == ""
    deny = args[args.index("--disallowedTools") + 1]
    # defense-in-depth deny-list still covers the dangerous built-ins + LSP.
    assert "Bash" in deny and "Read" in deny and "WebFetch" in deny and "LSP" in deny
    # System prompt + model wired through; JSON output.
    assert args[args.index("--system-prompt") + 1] == "You are a counter."
    assert args[args.index("--model") + 1] == "haiku"
    assert args[args.index("--output-format") + 1] == "json"


async def test_strips_anthropic_api_key_to_force_subscription(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    captured = _patch_invoke(monkeypatch, out=_result_array())

    await claude_code.run_completion("sys", [{"role": "user", "content": "hi"}], "haiku")

    assert "ANTHROPIC_API_KEY" not in captured["env"], \
        "API key must be stripped so Claude Code uses subscription auth"


async def test_omits_system_and_model_when_empty(monkeypatch):
    captured = _patch_invoke(monkeypatch, out=_result_array())
    await claude_code.run_completion("", [{"role": "user", "content": "hi"}], "")
    assert "--system-prompt" not in captured["args"]
    assert "--model" not in captured["args"]


# ---------------------------------------------------------------------------
# Output parsing + error handling
# ---------------------------------------------------------------------------

async def test_error_result_raises(monkeypatch):
    _patch_invoke(monkeypatch, out=_result_array("boom", is_error=True))
    with pytest.raises(claude_code.ClaudeCodeError, match="error result"):
        await claude_code.run_completion("s", [{"role": "user", "content": "x"}], "haiku")


async def test_nonzero_exit_raises(monkeypatch):
    _patch_invoke(monkeypatch, rc=1, err=b"kaboom")
    with pytest.raises(claude_code.ClaudeCodeError, match="exited 1"):
        await claude_code.run_completion("s", [{"role": "user", "content": "x"}], "haiku")


async def test_unparseable_output_raises(monkeypatch):
    _patch_invoke(monkeypatch, out=b"not json at all")
    with pytest.raises(claude_code.ClaudeCodeError, match="could not parse"):
        await claude_code.run_completion("s", [{"role": "user", "content": "x"}], "haiku")


async def test_missing_result_event_raises(monkeypatch):
    _patch_invoke(monkeypatch, out=json.dumps([{"type": "system"}]).encode())
    with pytest.raises(claude_code.ClaudeCodeError):
        await claude_code.run_completion("s", [{"role": "user", "content": "x"}], "haiku")


async def test_invoke_timeout_raises_claudecode_error(monkeypatch):
    """A subprocess timeout (the wait_for path) surfaces as a clean
    ClaudeCodeError, not a raw asyncio.TimeoutError."""
    import asyncio

    async def slow_invoke(args, env, cwd, timeout):
        raise asyncio.TimeoutError()
    monkeypatch.setattr(claude_code, "_invoke", slow_invoke)
    with pytest.raises(claude_code.ClaudeCodeError, match="timed out"):
        await claude_code.run_completion("s", [{"role": "user", "content": "x"}], "haiku")


async def test_invoke_missing_binary_raises_claudecode_error(monkeypatch):
    """`claude` not on PATH surfaces as a clear ClaudeCodeError naming the binary."""
    async def no_binary(args, env, cwd, timeout):
        raise FileNotFoundError(claude_code.CLAUDE_BIN)
    monkeypatch.setattr(claude_code, "_invoke", no_binary)
    with pytest.raises(claude_code.ClaudeCodeError, match="not found on PATH"):
        await claude_code.run_completion("s", [{"role": "user", "content": "x"}], "haiku")


# ---------------------------------------------------------------------------
# Delegation (executor) — command construction + the hard allow-list boundary
# ---------------------------------------------------------------------------

def _patch_invoke_deleg(monkeypatch, *, out=None):
    """Patch _invoke for delegation: also reads the --mcp-config file WHILE it
    still exists (inside run_delegated's temp dir) so tests can assert it."""
    out = out if out is not None else _result_array("counted")
    captured: dict = {}

    async def fake_invoke(args, env, cwd, timeout):
        captured["args"], captured["env"], captured["cwd"] = args, env, cwd
        with open(args[args.index("--mcp-config") + 1]) as f:
            captured["mcp_config"] = json.load(f)
        return 0, out, b""

    monkeypatch.setattr(claude_code, "_invoke", fake_invoke)
    return captured


def test_mcp_tool_name_maps_dotted_to_double_underscore():
    assert claude_code._mcp_tool_name("hello.count_to") == "mcp__hello__count_to"


async def test_delegation_builds_contained_command(monkeypatch):
    captured = _patch_invoke_deleg(monkeypatch)
    resp = await claude_code.run_delegated(
        "You are a counter.", [{"role": "user", "content": "count to 3"}], "haiku",
        allowed_tools=["hello.count_to"],
        mcp_servers={"hello": {"command": "python", "args": ["hello.py"]}})

    assert resp["choices"][0]["message"]["content"] == "counted"
    args = captured["args"]
    # allow-list maps to EXACTLY the recipe's mcp tool id, nothing else.
    assert args[args.index("--allowedTools") + 1] == "mcp__hello__count_to"
    # the built-in lockdown is kept: allow-list of built-ins = none, + deny-list.
    assert "--strict-mcp-config" in args
    assert args[args.index("--permission-mode") + 1] == "dontAsk"
    assert args[args.index("--tools") + 1] == ""        # no built-in tools at all
    assert "Bash" in args[args.index("--disallowedTools") + 1]
    # the --mcp-config contains ONLY the referenced server.
    assert captured["mcp_config"] == {
        "mcpServers": {"hello": {"command": "python", "args": ["hello.py"]}}}
    # delegation is multi-turn (NOT capped at 1 like the tool-less path).
    assert int(args[args.index("--max-turns") + 1]) > 1
    # system + model wired through.
    assert args[args.index("--system-prompt") + 1] == "You are a counter."
    assert args[args.index("--model") + 1] == "haiku"


async def test_delegation_allowlist_is_exact_boundary(monkeypatch):
    """The adversarial unit gate: woollama cannot widen Claude's grant beyond the
    recipe allow-list — N tools across M servers map to EXACTLY those ids, and the
    mcp-config has EXACTLY those servers."""
    captured = _patch_invoke_deleg(monkeypatch)
    await claude_code.run_delegated(
        "sys", [{"role": "user", "content": "go"}], "haiku",
        allowed_tools=["hello.count_to", "textops.word_count"],
        mcp_servers={"hello": {"command": "h", "args": []},
                     "textops": {"command": "t", "args": []}})
    allowed = captured["args"][captured["args"].index("--allowedTools") + 1].split(",")
    assert set(allowed) == {"mcp__hello__count_to", "mcp__textops__word_count"}
    assert set(captured["mcp_config"]["mcpServers"]) == {"hello", "textops"}


async def test_delegation_strips_harness_and_key_env(monkeypatch):
    """The child env is clean: subscription auth (no ANTHROPIC_API_KEY) AND no
    parent-harness leak (CLAUDECODE / CLAUDE_CODE_*) — the nested-contamination
    fix the spike surfaced."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leak")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent-sess")
    captured = _patch_invoke_deleg(monkeypatch)
    await claude_code.run_delegated(
        "s", [{"role": "user", "content": "x"}], "haiku",
        allowed_tools=["hello.count_to"],
        mcp_servers={"hello": {"command": "h", "args": []}})
    env = captured["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDECODE" not in env and "CLAUDE_CODE_SESSION_ID" not in env


async def test_tool_less_path_also_strips_harness_env(monkeypatch):
    """The env hardening covers the tool-less path too (same _child_env)."""
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    captured = _patch_invoke(monkeypatch, out=_result_array())
    await claude_code.run_completion("s", [{"role": "user", "content": "x"}], "haiku")
    assert "CLAUDECODE" not in captured["env"]
    assert "CLAUDE_CODE_ENTRYPOINT" not in captured["env"]
    # deferred-tool search disabled so MCP tools (delegation) load upfront once
    # --tools "" removes the built-in ToolSearch.
    assert captured["env"]["ENABLE_TOOL_SEARCH"] == "false"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def test_render_prompt_single_turn_verbatim():
    assert claude_code._render_prompt([{"role": "user", "content": "just this"}]) == "just this"


def test_render_prompt_multi_turn_role_prefixed():
    out = claude_code._render_prompt([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ])
    assert out == "user: a\nassistant: b\nuser: c"


def test_render_prompt_drops_system():
    out = claude_code._render_prompt([
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "kept"},
    ])
    assert out == "kept"
