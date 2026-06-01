"""Live integration tests against a real running Ollama and the bundled MCP
example servers.

Default-skipped (marked `integration`). Run with:

    pytest -m integration

Per-test skip if Ollama isn't reachable, so this is safe to run on any
machine — it just no-ops gracefully when the local backend isn't there.

Each test spawns its own woollama process on a random ephemeral port,
verifies the behavior end-to-end via the OpenAI Python SDK, then tears down."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

OLLAMA_URL = "http://localhost:11434"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _ollama_reachable() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.0).status_code == 200
    except Exception:
        return False


needs_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason="local Ollama not reachable at " + OLLAMA_URL,
)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


@pytest.fixture
def woollama_server(tmp_path):
    """Spawn `woollama` with a clean WOOLLAMA_CONFIG_DIR (so bundled defaults
    load) on a fresh ephemeral port. Yields the base URL. Tears down on exit.
    """
    port = _free_port()
    env = {
        **os.environ,
        "WOOLLAMA_ADDRESS": f"127.0.0.1:{port}",
        "WOOLLAMA_CONFIG_DIR": str(tmp_path),  # forces bundled defaults
        "XDG_RUNTIME_DIR": str(tmp_path),       # don't clobber the user's real addr-file
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "woollama"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        # Wait for the server to come up (max ~10s)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if httpx.get(f"{base_url}/v1/models", timeout=0.5).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            proc.terminate()
            raise RuntimeError(f"woollama didn't come up on {base_url}")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@needs_ollama
def test_models_endpoint_lists_ollama_and_recipes(woollama_server):
    """A live router with bundled defaults should surface both Ollama models
    and the bundled recipes via /v1/models."""
    r = httpx.get(f"{woollama_server}/v1/models", timeout=5)
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert any(i.startswith("ollama/") for i in ids), \
        "expected at least one ollama/ model; is Ollama returning models?"
    assert "woollama/streamer" in ids
    assert "woollama/textcounter" in ids


@needs_ollama
def test_passthrough_ollama_chat(woollama_server):
    """ollama/<X> goes straight through with no orchestration. Uses the
    smallest model we know is around to keep this snappy."""
    import openai
    c = openai.OpenAI(base_url=f"{woollama_server}/v1", api_key="not-required")
    # Pick whatever Ollama has — query /v1/models then pick the first ollama/ one
    models = httpx.get(f"{woollama_server}/v1/models", timeout=5).json()["data"]
    ollama_ids = [m["id"][len("ollama/"):] for m in models
                  if m["id"].startswith("ollama/")]
    if not ollama_ids:
        pytest.skip("no Ollama models available")
    r = c.chat.completions.create(
        model=f"ollama/{ollama_ids[0]}",
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        timeout=120,
    )
    assert r.choices[0].message.content
    # The model SHOULD reply with 'pong' but we don't insist — sufficient
    # to confirm we got an answer back at all
    assert r.choices[0].message.tool_calls is None


@needs_ollama
def test_orchestrated_recipe_hides_tool_loop_from_client(woollama_server):
    """The streamer recipe runs a chat-loop internally; the OpenAI client
    should see only the final answer, not the tool_calls."""
    import openai
    # Skip if qwen3 isn't available — the bundled recipe needs it
    models = httpx.get(f"{woollama_server}/v1/models", timeout=5).json()["data"]
    if not any("qwen3:14b-iq4xs" in m["id"] for m in models):
        pytest.skip("qwen3:14b-iq4xs not available; bundled recipe needs it")
    c = openai.OpenAI(base_url=f"{woollama_server}/v1", api_key="not-required")
    r = c.chat.completions.create(
        model="woollama/streamer",
        messages=[{"role": "user", "content": "Count to 3."}],
        timeout=180,
    )
    assert r.choices[0].message.content
    assert r.choices[0].message.tool_calls is None, \
        "client should not see internal tool_calls"


# ---------------------------------------------------------------------------
# woollama-as-MCP-server over real stdio (slice e)
# ---------------------------------------------------------------------------
#
# Unlike the in-memory unit tests in test_mcp_server.py, this drives a real
# `woollama mcp` subprocess over stdio with a real MCP client AND a *started*
# registry (the bundled hello + textops example servers spawn as their own
# subprocesses). That started registry is what the in-memory unit tests can't
# exercise — it's the only thing that proves registry.start_all() binds its
# connection-owning tasks to the same event loop that serves tool calls. No
# Ollama needed for the MCP surface itself, so this isn't gated on it.

async def test_mcp_stdio_surface_with_started_registry(tmp_path):
    """Spawn `woollama mcp` over stdio with bundled defaults; verify the MCP
    surface (capabilities, recipe prompts, the chat tool) AND that the real
    registry starts cleanly over stdio (hello + textops example servers)."""
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "woollama", "mcp"],
        env={**os.environ, "WOOLLAMA_CONFIG_DIR": str(tmp_path)},
        cwd=str(REPO_ROOT),
    )
    from fastmcp.exceptions import ToolError

    async with Client(transport) as c:
        caps = c.initialize_result.capabilities
        assert caps.tools is not None and caps.prompts is not None

        prompt_names = {p.name for p in await c.list_prompts()}
        assert {"streamer", "textcounter"} <= prompt_names

        tool_names = {t.name for t in await c.list_tools()}
        assert "chat" in tool_names

        # Connecting at all already proves the lifespan's registry.start_all()
        # didn't deadlock over stdio (the cross-loop hazard) and the server
        # came up clean (no banner corrupting the JSON-RPC stream). Calling the
        # tool proves it executes: an unknown recipe surfaces as a clean
        # ToolError rather than the transport silently dropping the request.
        # (This stops short of orchestration — recipes.get() short-circuits
        # before dispatch; the Ollama-gated test below drives the full loop.)
        with pytest.raises(ToolError, match="unknown recipe"):
            await c.call_tool("chat", {
                "recipe": "_no_such_recipe_",
                "messages": [{"role": "user", "content": "hi"}],
            })


@needs_ollama
async def test_mcp_stdio_chat_orchestrates_end_to_end(tmp_path):
    """Drive woollama as an MCP server over real stdio and run the `chat` tool
    through a full recipe orchestration (inferencer + tool dispatch). The MCP
    counterpart of test_orchestrated_recipe_hides_tool_loop_from_client — gives
    the MCP transport the same end-to-end parity the HTTP surface has."""
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    # The bundled streamer recipe needs qwen3:14b-iq4xs.
    if not _ollama_reachable() or "qwen3:14b-iq4xs" not in (
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2).text
    ):
        pytest.skip("qwen3:14b-iq4xs not available; bundled recipe needs it")

    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "woollama", "mcp"],
        env={**os.environ, "WOOLLAMA_CONFIG_DIR": str(tmp_path)},
        cwd=str(REPO_ROOT),
    )
    async with Client(transport) as c:
        result = await c.call_tool("chat", {
            "recipe": "streamer",
            "messages": [{"role": "user", "content": "Count to 3."}],
        })
    # Client sees only the final assistant string — the internal inferencer ↔
    # tool loop stays hidden, same contract as the OpenAI surface.
    assert isinstance(result.data, str) and result.data.strip()
