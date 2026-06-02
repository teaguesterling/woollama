"""Claude Code as a (tool-less) inference backend.

woollama routes a recipe whose inferencer is ``claude-code/<model>`` to the
local ``claude`` CLI in headless print mode, using the user's EXISTING Claude
Code auth (subscription/OAuth) — no ``ANTHROPIC_API_KEY`` required. It's a
keyless path to Claude, distinct from the OpenAI-compat HTTP inferencer seam
(vLLM/Together/Groq/anthropic-api), which is still unbuilt.

Scope: TOOL-LESS completions only. The recipe's system prompt shapes Claude,
the messages are the prompt, Claude returns one final answer. Letting Claude
Code run a recipe's MCP tools via its own agent loop (delegation) is a separate,
larger concept — an *executor*, not an inferencer — and a later slice; recipes
with a non-empty ``tools`` list routed here are rejected upstream in
``router.orchestrate``.

Why subprocess, not the Agent SDK: the SDK requires the ``claude`` CLI on PATH
anyway, so shelling out is fewer deps and trivially mockable (tests patch
``_invoke``).

Safety — keeping it genuinely tool-less (verified empirically against
v2.1.160): ``--permission-mode dontAsk`` auto-DENIES tools that would otherwise
prompt, but it still auto-RUNS read-only Bash (a live test showed ``echo``
executing). So we ALSO ``--disallowedTools`` the exec/file/network/subagent
vectors — Bash above all — and ``--strict-mcp-config`` to load zero MCP servers,
and run in a neutral temp cwd so we don't inherit the host's CLAUDE.md /
settings / plugins. The opt-in live test (tests/test_integration.py,
``@needs_claude_code``) verifies a Bash attempt is refused.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile

log = logging.getLogger("woollama.claude_code")

# Override the binary (e.g. an absolute path) via this env var.
CLAUDE_BIN = os.environ.get("WOOLLAMA_CLAUDE_BIN", "claude")

# Defense-in-depth tool lockdown. dontAsk already auto-denies tools that would
# prompt, but read-only Bash slips through — so deny it explicitly, plus the
# other obvious leak/exec vectors. Tools NOT listed are still auto-denied by
# dontAsk; this list is belt-and-suspenders for the dangerous categories.
_DENY_TOOLS = ("Bash,Read,Write,Edit,NotebookEdit,WebFetch,WebSearch,"
               "Glob,Grep,Task")


class ClaudeCodeError(RuntimeError):
    """The Claude Code backend failed: spawn error, non-zero exit, an error
    result, a timeout, or unparseable output."""


def _render_prompt(user_msgs: list[dict]) -> str:
    """Flatten OpenAI messages into a single prompt for ``claude -p``.

    A single user turn → its content verbatim; multiple/mixed turns →
    role-prefixed lines (a v1 simplification — full multi-turn fidelity via
    ``--input-format stream-json`` is a later refinement). System messages are
    dropped here: the recipe's system prompt is passed via ``--system-prompt``.
    """
    msgs = [m for m in user_msgs if m.get("role") != "system"]
    if len(msgs) == 1:
        return str(msgs[0].get("content") or "")
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content') or ''}"
                     for m in msgs)


def _extract_result(stdout: str) -> tuple[str, bool]:
    """Parse ``claude -p --output-format json`` output. This version emits a
    JSON ARRAY of events; the final assistant text is the ``type == "result"``
    event's ``result`` field. Returns ``(text, is_error)``."""
    data = json.loads(stdout)
    events = data if isinstance(data, list) else [data]
    for ev in reversed(events):
        if isinstance(ev, dict) and ev.get("type") == "result":
            return str(ev.get("result") or ""), bool(ev.get("is_error"))
    raise ClaudeCodeError("no 'result' event in claude output")


async def _invoke(args: list[str], env: dict, cwd: str,
                  timeout: float) -> tuple[int, bytes, bytes]:
    """Run the subprocess; return (returncode, stdout, stderr). The mock seam:
    tests patch this so no real ``claude`` is spawned."""
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, cwd=cwd, env=env)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode, out, err


async def run_completion(system: str, user_msgs: list[dict], model: str,
                         *, timeout: float = 180.0) -> dict:
    """Run a one-shot, tool-less Claude completion via ``claude -p`` and return
    an OpenAI-shaped chat-completions response dict. Raises ``ClaudeCodeError``
    on any failure."""
    prompt = _render_prompt(user_msgs)
    args = [CLAUDE_BIN, "-p", prompt,
            "--output-format", "json",
            "--max-turns", "1",
            "--strict-mcp-config",                 # zero MCP servers
            "--permission-mode", "dontAsk",        # non-interactive (no hang)
            "--disallowedTools", _DENY_TOOLS]      # ...and genuinely tool-less
    if system:
        args += ["--system-prompt", system]
    if model:
        args += ["--model", model]

    # Force subscription auth (the keyless point): an ANTHROPIC_API_KEY in the
    # child env would silently switch Claude Code to API billing.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    with tempfile.TemporaryDirectory() as cwd:
        try:
            rc, out, err = await _invoke(args, env, cwd, timeout)
        except asyncio.TimeoutError as e:
            raise ClaudeCodeError(f"claude timed out after {timeout}s") from e
        except FileNotFoundError as e:
            raise ClaudeCodeError(f"`{CLAUDE_BIN}` not found on PATH") from e

    if rc != 0:
        raise ClaudeCodeError(
            f"claude exited {rc}: {err.decode('utf-8', 'replace')[:300]}")
    try:
        text, is_error = _extract_result(out.decode("utf-8", "replace"))
    except json.JSONDecodeError as e:
        raise ClaudeCodeError(f"could not parse claude output: {e}") from e
    if is_error:
        raise ClaudeCodeError(f"claude returned an error result: {text[:300]}")

    return {
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": text},
        }],
    }
