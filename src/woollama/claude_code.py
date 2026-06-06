"""Claude Code as an inference backend — tool-less completions AND delegation.

woollama routes a recipe whose inferencer is ``claude-code/<model>`` to the
local ``claude`` CLI in headless print mode, using the user's EXISTING Claude
Code auth (subscription/OAuth) — no ``ANTHROPIC_API_KEY`` required. It's a
keyless path to Claude, distinct from the OpenAI-compat HTTP inferencer seam
(vLLM/Together/Groq/anthropic-api), which is still unbuilt.

Two modes:

* **Tool-less completion** (``run_completion`` / ``run_resumable``): the recipe's
  system prompt shapes Claude, the messages are the prompt, Claude returns one
  final answer. Recipes with an empty ``tools`` list.
* **Delegation / executor** (``run_delegated``): Claude OWNS the agentic loop and
  calls the recipe's allow-listed MCP tools itself; woollama returns only the
  final answer. Recipes with a non-empty ``tools`` list. This is an *executor*,
  not an inferencer.

Why subprocess, not the Agent SDK: the SDK requires the ``claude`` CLI on PATH
anyway, so shelling out is fewer deps and trivially mockable (tests patch
``_invoke``).

Safety. ``--permission-mode dontAsk`` auto-DENIES any tool not pre-approved (a
HARD deny, recorded in the result's ``permission_denials`` — verified live), but
it still auto-RUNS read-only Bash, so we ALSO ``--disallowedTools`` the exec/
file/network/subagent vectors (Bash above all). We run in a neutral temp cwd and
strip the child env (see ``_child_env``) so we don't inherit the host's
CLAUDE.md / settings / plugins / nested-harness. For tool-less mode
``--strict-mcp-config`` loads ZERO MCP servers. For delegation we write a
per-recipe ``--mcp-config`` with ONLY the servers the allow-list references and
pass ``--allowedTools`` listing ONLY those tools — so the recipe allow-list stays
a hard boundary even though Claude drives the loop (defense-in-depth: config
containment AND allow-list AND the built-in lockdown). Delegation only ADDS the
recipe's MCP tools; it removes none of the above. Opt-in live tests
(tests/test_integration.py, ``@needs_claude_code``, run in a PLAIN terminal — a
nested child inherits the parent harness) verify both a Bash refusal and that an
out-of-list tool is denied in delegation mode.
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


def _child_env() -> dict:
    """Environment for the ``claude`` child process. Strips:

    * ``ANTHROPIC_API_KEY`` — force subscription/OAuth auth (the keyless point);
      a key in the child would silently switch Claude Code to API billing.
    * ``CLAUDECODE`` / ``CLAUDE_CODE_*`` — the parent harness vars. When woollama
      itself runs INSIDE a Claude Code session, leaving these set leaks the
      parent harness into the child (its meta-tools / nested deferred-tool mode —
      observed contaminating the delegation spike). Stripping them gives a clean
      child regardless of where woollama runs.
    """
    return {k: v for k, v in os.environ.items()
            if k != "ANTHROPIC_API_KEY"
            and k != "CLAUDECODE"
            and not k.startswith("CLAUDE_CODE")}


def _mcp_tool_name(namespaced: str) -> str:
    """Map a recipe's ``<server>.<tool>`` to the id Claude Code exposes for a
    tool from a ``--mcp-config`` server: ``mcp__<server>__<tool>`` (the clean,
    no-dot naming verified in the delegation spike)."""
    server, _, tool = namespaced.partition(".")
    return f"mcp__{server}__{tool}"


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


def _extract(stdout: str) -> tuple[str, bool, str | None]:
    """Parse ``claude -p --output-format json`` (a JSON ARRAY of events). The
    final assistant text + error flag come from the ``type == "result"`` event,
    which also carries the ``session_id`` (verified live, v2.1.163) — that id is
    what lets the claude-resume backend continue the session. Returns
    ``(text, is_error, session_id)``."""
    data = json.loads(stdout)
    events = data if isinstance(data, list) else [data]
    for ev in reversed(events):
        if isinstance(ev, dict) and ev.get("type") == "result":
            return (str(ev.get("result") or ""), bool(ev.get("is_error")),
                    ev.get("session_id"))
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


def _build_args(prompt: str, system: str, model: str,
                resume: str | None) -> list[str]:
    args = [CLAUDE_BIN, "-p", prompt,
            "--output-format", "json",
            "--max-turns", "1",
            "--strict-mcp-config",                 # zero MCP servers
            "--permission-mode", "dontAsk",        # non-interactive (no hang)
            "--disallowedTools", _DENY_TOOLS]      # ...and genuinely tool-less
    if resume:
        args += ["--resume", resume]               # continue an existing session
    # The system prompt is set when STARTING a session; --resume carries it
    # forward, so we don't (and shouldn't) re-send it on continuation turns.
    if system and not resume:
        args += ["--system-prompt", system]
    if model:
        args += ["--model", model]
    return args


async def _invoke_and_parse(args: list[str], env: dict, cwd: str,
                            timeout: float) -> tuple[str, str | None]:
    """Run claude in `cwd` and parse → (final_text, session_id). Raises
    ClaudeCodeError on spawn/exit/parse/error-result."""
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
        text, is_error, sid = _extract(out.decode("utf-8", "replace"))
    except json.JSONDecodeError as e:
        raise ClaudeCodeError(f"could not parse claude output: {e}") from e
    if is_error:
        raise ClaudeCodeError(f"claude returned an error result: {text[:300]}")
    return text, sid


async def _run(prompt: str, system: str, model: str, timeout: float,
               *, resume: str | None = None,
               cwd: str | None = None) -> tuple[str, str | None]:
    """Shared invoke+parse core. `cwd` is load-bearing for resume: Claude Code
    scopes sessions BY PROJECT (cwd), so all turns of one conversation must run
    in the SAME directory or `--resume` fails with "No conversation found". A
    one-shot completion (`cwd=None`) gets a throwaway temp dir; a resumable
    conversation passes its own stable workdir."""
    args = _build_args(prompt, system, model, resume)
    env = _child_env()
    if cwd is not None:
        return await _invoke_and_parse(args, env, cwd, timeout)
    with tempfile.TemporaryDirectory() as tmp:
        return await _invoke_and_parse(args, env, tmp, timeout)


def _as_openai(text: str) -> dict:
    return {
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": text},
        }],
    }


async def run_completion(system: str, user_msgs: list[dict], model: str,
                         *, timeout: float = 180.0) -> dict:
    """Run a one-shot, tool-less Claude completion via ``claude -p`` and return
    an OpenAI-shaped chat-completions response dict. Raises ``ClaudeCodeError``
    on any failure."""
    text, _ = await _run(_render_prompt(user_msgs), system, model, timeout)
    return _as_openai(text)


async def run_resumable(system: str, user_msgs: list[dict], model: str,
                        *, session_id: str | None = None, cwd: str,
                        timeout: float = 180.0) -> tuple[dict, str | None]:
    """One tool-less turn that PARTICIPATES IN A SESSION — the claude-resume
    backend (slice conv-1b). ``session_id=None`` starts a new session;
    otherwise ``--resume`` it. ``cwd`` MUST be stable across a conversation's
    turns (Claude scopes sessions by project dir — see `_run`). Returns
    ``(OpenAI-shaped dict, session_id)``; the captured ``session_id`` is the
    handle's backing id, stored so the next turn can resume (resume reports the
    SAME id)."""
    text, sid = await _run(_render_prompt(user_msgs), system, model, timeout,
                           resume=session_id, cwd=cwd)
    return _as_openai(text), sid


def _build_delegate_args(prompt: str, system: str, model: str,
                         mcp_config_path: str, allowed: list[str],
                         max_turns: int) -> list[str]:
    """argv for a delegated (executor) turn. Reuses the slice-i lockdown verbatim
    (``dontAsk`` + ``_DENY_TOOLS`` + ``--strict-mcp-config``) and ADDS the
    per-recipe ``--mcp-config`` plus ``--allowedTools`` (only the recipe's tools).
    No ``--max-turns 1``: delegation is a multi-turn agentic loop, capped at
    ``max_turns`` (a cost + safety bound)."""
    args = [CLAUDE_BIN, "-p", prompt,
            "--output-format", "json",
            "--max-turns", str(max_turns),
            "--mcp-config", mcp_config_path,
            "--strict-mcp-config",                 # ONLY the config we write loads
            "--permission-mode", "dontAsk",        # hard-denies anything not allow-listed
            "--disallowedTools", _DENY_TOOLS,      # built-in exec/file/net lockdown
            "--allowedTools", ",".join(allowed)]   # ONLY the recipe's MCP tools
    if system:
        args += ["--system-prompt", system]
    if model:
        args += ["--model", model]
    return args


async def run_delegated(system: str, user_msgs: list[dict], model: str, *,
                        allowed_tools: list[str], mcp_servers: dict[str, dict],
                        max_turns: int = 8, timeout: float = 300.0) -> dict:
    """Delegated EXECUTOR turn: hand Claude Code the recipe's system prompt and
    its allow-listed MCP tools and let Claude run the agentic loop itself,
    returning an OpenAI-shaped chat-completions dict with the final answer.

    The recipe allow-list stays a HARD boundary even though Claude drives:
    ``--permission-mode dontAsk`` denies any tool not in ``--allowedTools``
    (verified via spike), the ``--mcp-config`` we write contains ONLY the servers
    ``allowed_tools`` references (config containment), and the built-in lockdown
    is kept. Defense-in-depth across three independent layers.

    ``allowed_tools``: the recipe's ``<server>.<tool>`` names.
    ``mcp_servers``: ``{server_name: {"command", "args"}}`` for the referenced
    servers only (the caller filters config down to what the allow-list needs).

    Raises ``ClaudeCodeError`` on any failure."""
    allowed = [_mcp_tool_name(t) for t in allowed_tools]
    env = _child_env()
    # Neutral temp cwd (no inherited CLAUDE.md/settings); the mcp config lives
    # inside it under a non-".mcp.json" name so it isn't auto-discovered —
    # --strict-mcp-config means only the file we pass loads anyway.
    with tempfile.TemporaryDirectory() as cwd:
        cfg_path = os.path.join(cwd, "delegate-mcp.json")
        with open(cfg_path, "w") as f:
            json.dump({"mcpServers": mcp_servers}, f)
        args = _build_delegate_args(_render_prompt(user_msgs), system, model,
                                    cfg_path, allowed, max_turns)
        text, _ = await _invoke_and_parse(args, env, cwd, timeout)
    return _as_openai(text)
