#!/usr/bin/env python3
"""woollama status demo — boots the real server and VALIDATES each shipped
capability against live Ollama, printing a pass/fail report.

What it exercises:
  * dual binding      — Unix socket (mode 0600) AND loopback TCP, same app
  * discovery         — /v1/models, /v1/tools (the latter over the UDS)
  * streaming-1       — passthrough `ollama/<model>` with stream:true (verbatim SSE)
  * streaming-2       — `woollama/<recipe>` orchestration streamed; tool loop hidden
  * streaming-3       — MCP `chat` tool emits ctx.info progress over /mcp

Run:  uv run --extra dev python examples/status_demo.py
Needs a reachable Ollama with qwen3:14b-iq4xs (the bundled recipe's model).
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
MODEL = "qwen3:14b-iq4xs"
OK, BAD = "\033[32m✓\033[0m", "\033[31m✗\033[0m"
results: list[tuple[bool, str]] = []


def check(cond: bool, label: str) -> bool:
    results.append((bool(cond), label))
    print(f"  {OK if cond else BAD} {label}")
    return bool(cond)


def banner(title: str) -> None:
    print(f"\n\033[1m── {title}\033[0m")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def parse_sse(resp):
    """Yield parsed `data:` JSON objects (and the literal '[DONE]') from an
    httpx streaming response, as they arrive."""
    async for line in resp.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            yield "[DONE]"
        else:
            yield json.loads(data)


async def main() -> int:
    rt = tempfile.mkdtemp(prefix="woollama-demo-")
    port = free_port()
    sock = os.path.join(rt, "woollama.sock")
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "WOOLLAMA_ADDRESS": f"127.0.0.1:{port}",
           "WOOLLAMA_CONFIG_DIR": rt, "XDG_RUNTIME_DIR": rt}

    print(f"booting woollama on {base}  (+ unix socket {sock})")
    proc = subprocess.Popen([sys.executable, "-m", "woollama"], cwd=REPO, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # ---- wait for readiness -------------------------------------------
        for _ in range(100):
            try:
                if httpx.get(f"{base}/v1/models", timeout=0.5).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            print("server did not come up")
            return 1

        ollama_ok = False
        try:
            ollama_ok = httpx.get("http://localhost:11434/api/tags",
                                  timeout=2).status_code == 200
        except Exception:
            pass

        # ---- 1. dual binding ----------------------------------------------
        banner("Binding — Unix socket alongside TCP loopback")
        check(os.path.exists(sock), f"unix socket exists: {sock}")
        mode = stat.S_IMODE(os.stat(sock).st_mode)
        check(mode == 0o600, f"socket mode is 0600 (got {oct(mode)})")
        check(stat.S_ISSOCK(os.stat(sock).st_mode), "it is a real AF_UNIX socket")

        # ---- 2. discovery (TCP + UDS) -------------------------------------
        banner("Discovery — /v1/models (TCP) and /v1/tools (UDS)")
        models = httpx.get(f"{base}/v1/models", timeout=5).json()["data"]
        ids = [m["id"] for m in models]
        print("  models:", ", ".join(ids))
        check(any(i.startswith("woollama/") for i in ids), "recipes advertised as woollama/<recipe>")
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=sock)) as uc:
            tools = (await uc.get("http://localhost/v1/tools", timeout=5)).json()["tools"]
        print("  tools (over UDS):", ", ".join(tools))
        check(any("." in t for t in tools), "namespaced downstream tools served over the Unix socket")

        if not ollama_ok:
            banner("Ollama unreachable — skipping the live streaming checks")
            print("  (start Ollama with the bundled model to run streaming-1/2/3)")
            return report()

        have_model = any(MODEL in i for i in ids)
        if not have_model:
            banner(f"{MODEL} not pulled — skipping live streaming checks")
            return report()

        # ---- 3. streaming-1: passthrough SSE ------------------------------
        banner("streaming-1 — passthrough ollama/<model> with stream:true")
        chunks = 0
        text = ""
        async with httpx.AsyncClient(timeout=120) as c:
            async with c.stream("POST", f"{base}/v1/chat/completions", json={
                "model": f"ollama/{MODEL}", "stream": True,
                "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
            }) as resp:
                async for obj in parse_sse(resp):
                    if obj == "[DONE]":
                        break
                    chunks += 1
                    text += obj["choices"][0].get("delta", {}).get("content", "")
        print(f"  received {chunks} SSE chunks; assembled: {text.strip()!r}")
        check(chunks > 1, "upstream stream relayed as multiple SSE chunks")
        check(bool(text.strip()), "passthrough produced content")

        # ---- 4. streaming-2: orchestration SSE, tool loop hidden ----------
        banner("streaming-2 — woollama/streamer streamed; tool loop hidden")
        print("  prompt: 'Count to 3.'  (recipe calls hello.count_to internally)")
        t0 = time.monotonic()
        deltas, finishes, saw_tool, first_at = [], [], False, None
        async with httpx.AsyncClient(timeout=180) as c:
            async with c.stream("POST", f"{base}/v1/chat/completions", json={
                "model": "woollama/streamer", "stream": True,
                "messages": [{"role": "user", "content": "Count to 3."}],
            }) as resp:
                async for obj in parse_sse(resp):
                    if obj == "[DONE]":
                        break
                    ch = obj["choices"][0]
                    piece = ch.get("delta", {}).get("content")
                    if piece:
                        if first_at is None:
                            first_at = time.monotonic() - t0
                        deltas.append(piece)
                    if ch.get("delta", {}).get("tool_calls"):
                        saw_tool = True
                    if ch.get("finish_reason"):
                        finishes.append(ch["finish_reason"])
        answer = "".join(deltas)
        print(f"  first token after {first_at:.1f}s; streamed answer: {answer.strip()!r}")
        check(bool(answer.strip()), "final answer arrived as streamed deltas")
        check(not saw_tool, "no tool_call JSON leaked to the client (loop hidden)")
        check(finishes == ["stop"], f"exactly one terminator (got {finishes})")

        # ---- 5. streaming-3: MCP chat progress over /mcp ------------------
        banner("streaming-3 — MCP chat tool emits live progress over /mcp")
        from fastmcp import Client
        logs: list[str] = []

        async def log_handler(params):
            d = params.data
            logs.append(d["msg"] if isinstance(d, dict) and "msg" in d else str(d))

        async with Client(f"{base}/mcp", log_handler=log_handler) as mc:
            tool_names = {t.name for t in await mc.list_tools()}
            res = await mc.call_tool("chat", {
                "recipe": "streamer",
                "messages": [{"role": "user", "content": "Count to 3."}],
            })
        print("  progress notifications received during the call:")
        for line in logs:
            print(f"      • {line}")
        print(f"  tool result: {str(res.data).strip()!r}")
        check("chat" in tool_names, "chat verb advertised on the MCP surface")
        check(any("→" in m for m in logs), "tool-call progress surfaced via ctx.info")
        check(any("count_to" in m for m in logs), "the hidden hello.count_to dispatch was reported")
        check(bool(str(res.data).strip()), "MCP chat returned the final answer")

        return report()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def report() -> int:
    banner("Summary")
    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
    for ok, label in results:
        if not ok:
            print(f"  {BAD} {label}")
    print(f"\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
