#!/usr/bin/env python
"""routing_demo.py — watch woollama route requests, live.

Spins up woollama (both its OpenAI HTTP surface and its MCP stdio surface) with
the bundled default config (the hello + textops example servers) and walks
through every routing activity, printing what goes where:

  1. Discovery        — /v1/models and /v1/tools (the union across providers)
  2. Passthrough      — ollama/<model> straight to Ollama (needs Ollama)
  3. Orchestration    — woollama/textcounter: ONE chat using tools from TWO
                        providers (textops + hello), proxied across two sessions
                        (needs Ollama)
  4. MCP aggregator   — connect AS an MCP client; list recipes-as-prompts and
                        the chat verb + every re-exported downstream tool, then
                        call hello.count_to and textops.word_count directly
                        (no Ollama needed)
  5. Rejections       — the things that must NOT work

Run:    python examples/routing_demo.py
Ollama: parts 2 and 3 are skipped (clearly) if Ollama isn't reachable; the
        rest runs anywhere.
"""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

import httpx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLLAMA_URL = os.environ.get("WOOLLAMA_OLLAMA_URL", "http://localhost:11434")


def hdr(title: str) -> None:
    print(f"\n{'═' * 72}\n  {title}\n{'═' * 72}")


def ollama_up() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1).status_code == 200
    except Exception:
        return False


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def woollama_http(config_dir: str):
    """Spawn the OpenAI HTTP surface on an ephemeral port; yield its base URL."""
    port = _free_port()
    env = {**os.environ, "WOOLLAMA_ADDRESS": f"127.0.0.1:{port}",
           "WOOLLAMA_CONFIG_DIR": config_dir, "XDG_RUNTIME_DIR": config_dir}
    proc = subprocess.Popen([sys.executable, "-m", "woollama"], cwd=REPO_ROOT,
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(50):
            try:
                if httpx.get(f"{base}/v1/models", timeout=0.5).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("woollama HTTP didn't come up")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def http_demo(config_dir: str) -> None:
    with woollama_http(config_dir) as base:
        c = httpx.Client(base_url=base, timeout=180)

        hdr("1. Discovery")
        models = c.get("/v1/models").json()["data"]
        print("  /v1/models:")
        for m in models:
            print(f"    {m['id']:32}  (owned_by={m['owned_by']})")
        tools = c.get("/v1/tools").json()["tools"]
        print(f"  /v1/tools (namespaced across providers): {tools}")

        if ollama_up():
            ollama_ids = [m["id"][len("ollama/"):] for m in models
                          if m["id"].startswith("ollama/")]
            hdr("2. Passthrough — ollama/<model> (no tools, straight through)")
            if ollama_ids:
                r = c.post("/v1/chat/completions", json={
                    "model": f"ollama/{ollama_ids[0]}",
                    "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
                }).json()
                print(f"  ollama/{ollama_ids[0]} → {r['choices'][0]['message']['content']!r}")

            hdr("3. Orchestration — woollama/textcounter (TWO providers, one chat)")
            print("  recipe textcounter allow-lists: textops.word_count + hello.count_to")
            print("  → model calls word_count (textops session), then count_to (hello session)")
            r = c.post("/v1/chat/completions", json={
                "model": "woollama/textcounter",
                "messages": [{"role": "user",
                              "content": "Count the words in: the quick brown fox"}],
            }).json()
            msg = r["choices"][0]["message"]
            print(f"  final answer (tool loop hidden): {msg['content']!r}")
            print(f"  tool_calls leaked to client? {msg.get('tool_calls')}")
        else:
            hdr("2-3. Passthrough + orchestration — SKIPPED (Ollama not reachable)")
            print(f"  start Ollama at {OLLAMA_URL} to see these.")

        hdr("5a. Rejections (HTTP) — must NOT work")
        for model in ("bogus/x", "woollama/does-not-exist"):
            r = c.post("/v1/chat/completions",
                       json={"model": model, "messages": []})
            print(f"  model={model!r:28} → HTTP {r.status_code}  {r.json()['error']['type']}")


async def mcp_demo(config_dir: str) -> None:
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    from fastmcp.exceptions import ToolError

    transport = StdioTransport(
        command=sys.executable, args=["-m", "woollama", "mcp"],
        env={**os.environ, "WOOLLAMA_CONFIG_DIR": config_dir}, cwd=REPO_ROOT)

    hdr("4. MCP aggregator — woollama AS an MCP server (no Ollama needed)")
    async with Client(transport) as client:
        prompts = await client.list_prompts()
        print(f"  prompts (recipes): {[p.name for p in prompts]}")
        tools = sorted(t.name for t in await client.list_tools())
        print(f"  tools (chat verb + re-exported downstream, namespaced): {tools}")

        print("\n  Direct proxy calls — each routes to its own session:")
        r1 = await client.call_tool("hello.count_to", {"n": 3})
        print(f"    hello.count_to(n=3)            → {r1.data}")
        r2 = await client.call_tool("textops.word_count", {"text": "the quick brown fox"})
        print(f"    textops.word_count('...fox')   → {r2.data}")

        hdr("5b. Rejections (MCP) — must NOT work")
        try:
            await client.call_tool("chat", {"recipe": "ghost", "messages": []})
        except ToolError as e:
            print(f"  chat(recipe='ghost')           → refused: {e}")
        try:
            await client.call_tool("hello.no_such_tool", {})
        except ToolError as e:
            print(f"  hello.no_such_tool()           → refused: {e}")

    print("\n  NOTE: the recipe allow-list boundary (a recipe cannot dispatch a")
    print("  tool outside its declared list, even if that session is connected)")
    print("  needs a model emitting an out-of-list call — see the deterministic")
    print("  test_routing.py::test_reject_tool_outside_recipe_allowlist.")


def main() -> int:
    print("woollama routing demo — bundled defaults (hello + textops servers)")
    print(f"Ollama at {OLLAMA_URL}: {'reachable' if ollama_up() else 'NOT reachable (parts 2-3 skip)'}")
    with tempfile.TemporaryDirectory() as config_dir:  # forces bundled defaults
        http_demo(config_dir)
        asyncio.run(mcp_demo(config_dir))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
