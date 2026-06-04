#!/usr/bin/env python3
"""Rigorous validation of woollama's ROUTING — and an interactive report.

woollama routes on three axes (docs/architecture.md): MODELS (`<provider>/<name>`
→ which backend / recipe), TOOLS (`<server>.<tool>` → which downstream MCP
session), and the two SURFACES (OpenAI HTTP + MCP) that share one core. This
harness probes each, mixing:

  * DETERMINISTIC in-process probes — a scripted inferencer + recording
    ServerManagers give *definitive attribution* (which session a call hit),
    with no model in the loop.
  * LIVE probes — against a freshly-booted `woollama` over real Ollama. Live
    HTTP orchestration hides tool results from the client BY DESIGN, so it can't
    prove fan-out on its own; we use the MCP `chat` tool's progress
    notifications (slice streaming-3) as the live routing instrument.

It does NOT hide failures: an unexpected exception is recorded as FAIL with the
real error, model-dependent probes are run repeatedly and reported as a ratio,
and skips state why. Output: routing_validation_report.html (+ .json).

Run:  uv run --extra dev python examples/routing_validation.py
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx

REPO = Path(__file__).resolve().parent.parent
MODEL = "qwen3:14b-iq4xs"
OUT_HTML = REPO / "routing_validation_report.html"
OUT_JSON = REPO / "routing_validation_results.json"


@dataclass
class Probe:
    id: str
    axis: str
    intent: str
    proves: str       # attribution | integration | negative | config
    mode: str         # deterministic | live | skip
    status: str = "FAIL"   # PASS | FAIL | PARTIAL | SKIP
    request: str = ""
    decision: str = ""     # where the request was routed / the decision made
    outcome: str = ""
    commentary: str = ""
    detail: str = ""       # raw payloads / errors (monospace)


probes: list[Probe] = []


def emit(p: Probe) -> None:
    probes.append(p)
    icon = {"PASS": "✓", "FAIL": "✗", "PARTIAL": "≈", "SKIP": "–"}[p.status]
    print(f"  [{icon}] {p.id:5} {p.status:7} {p.intent}")


async def run_probe(p: Probe, body) -> None:
    """Execute `body(p)` which mutates p; an uncaught exception => FAIL with the
    real traceback (we do NOT swallow failures)."""
    try:
        await body(p)
    except Exception as e:  # noqa: BLE001 — the whole point is to surface it
        p.status = "FAIL"
        p.outcome = p.outcome or f"unexpected {type(e).__name__}: {e}"
        p.detail = (p.detail + "\n" + traceback.format_exc()).strip()
    emit(p)


# ---------------------------------------------------------------------------
# Fakes for the deterministic probes (scripted inferencer + recording sessions)
# ---------------------------------------------------------------------------

def scripted_httpx(turns: list[dict]):
    """A fake httpx.AsyncClient whose .post() returns the next scripted OpenAI
    response (non-streaming orchestration path). Restore httpx.AsyncClient after."""
    script = list(turns)

    class _Resp:
        def __init__(self, payload): self._p = payload
        @property
        def status_code(self): return 200
        def json(self): return self._p

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def get(self, *a, **k): return _Resp({})
        async def post(self, *a, **k): return _Resp(script.pop(0))

    return _Client


def tool_call_turn(name, args, cid):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def final_turn(text):
    return {"choices": [{"message": {"content": text}}]}


def recording_manager(server, tool):
    from woollama.manager import ServerManager
    calls: list = []
    mgr = ServerManager(server, "echo", [])
    mgr.tools = [SimpleNamespace(name=tool, description=f"{server}.{tool}",
                                 inputSchema={"type": "object", "properties": {}})]

    async def _call(bare, args):
        calls.append((bare, args))
        return SimpleNamespace(content=[SimpleNamespace(text=f"{server}.{bare} ok")],
                               isError=False)
    mgr.call_tool = _call  # type: ignore[method-assign]
    return mgr, calls


def two_session_registry():
    from woollama.manager import Registry
    reg = Registry()
    hello, hello_calls = recording_manager("hello", "count_to")
    textops, textops_calls = recording_manager("textops", "word_count")
    reg.add(hello)
    reg.add(textops)
    return reg, hello_calls, textops_calls


# ===========================================================================
# DETERMINISTIC probes (in-process — definitive attribution, no model)
# ===========================================================================

async def deterministic_probes() -> None:
    import woollama.router as router

    print("\n── DETERMINISTIC (in-process; scripted inferencer + recording sessions)")

    # B-det: cross-session fan-out — the headline, proven definitively.
    async def b_det(p: Probe):
        reg, hello_calls, textops_calls = two_session_registry()
        recipe = {"inferencer": "ollama/x", "system": "fan out",
                  "tools": ["textops.word_count", "hello.count_to"]}
        p.request = ('recipe allow-list = [textops.word_count, hello.count_to]; '
                     'scripted model calls word_count then count_to')
        orig = httpx.AsyncClient
        httpx.AsyncClient = scripted_httpx([
            tool_call_turn("textops.word_count", {"text": "a b c"}, "t1"),
            tool_call_turn("hello.count_to", {"n": 3}, "t2"),
            final_turn("done"),
        ])
        try:
            await router.orchestrate(recipe, [{"role": "user", "content": "go"}], reg)
        finally:
            httpx.AsyncClient = orig
        p.decision = f"textops session got {textops_calls}; hello session got {hello_calls}"
        ok = (textops_calls == [("word_count", {"text": "a b c"})]
              and hello_calls == [("count_to", {"n": 3})])
        p.status = "PASS" if ok else "FAIL"
        p.outcome = ("each call landed on its OWN session, routed by namespace prefix"
                     if ok else "MISROUTED — calls did not match expected sessions")
        p.commentary = ("Definitive attribution: the recording managers record exactly "
                        "what each session received, so this proves the RIGHT call hit "
                        "the RIGHT session — not merely that the stack ran.")
        p.detail = f"textops_calls={textops_calls}\nhello_calls={hello_calls}"

    await run_probe(Probe("B-det", "B. Tool routing across sessions",
                          "Cross-session fan-out routes each tool to its owning session",
                          "attribution", "deterministic"), b_det)

    # C-det: the allow-list boundary — an out-of-list call must NOT be routed.
    async def c_det(p: Probe):
        reg, hello_calls, textops_calls = two_session_registry()
        recipe = {"inferencer": "ollama/x", "system": "locked",
                  "tools": ["hello.count_to"]}            # textops NOT allowed
        p.request = ('recipe allow-list = [hello.count_to] only; scripted model '
                     'tries textops.word_count (out of list)')
        orig = httpx.AsyncClient
        httpx.AsyncClient = scripted_httpx([
            tool_call_turn("textops.word_count", {"text": "x"}, "t1"),
            final_turn("recovered"),
        ])
        try:
            resp = await router.orchestrate(recipe, [{"role": "user", "content": "go"}], reg)
        finally:
            httpx.AsyncClient = orig
        refused = textops_calls == [] and hello_calls == []
        p.decision = (f"textops session got {textops_calls} (must be empty); "
                      "out-of-list call was refused, not dispatched")
        p.status = "PASS" if refused else "FAIL"
        p.outcome = ("boundary held: the forbidden tool was NOT routed to textops"
                     if refused else "LEAK — a forbidden tool reached a session")
        p.commentary = ("The allow-list is a security boundary, not a hint. Even though "
                        "the model emitted a tool_call for a tool on a real, connected "
                        "session, the recipe was never granted it, so dispatch is refused "
                        "and the loop recovers to a final answer.")
        p.detail = (f"textops_calls={textops_calls}\nhello_calls={hello_calls}\n"
                    f"final={resp['choices'][0]['message'].get('content')!r}")

    await run_probe(Probe("C-det", "C. Allow-list boundary",
                          "An out-of-allow-list tool_call is refused, never routed",
                          "negative", "deterministic"), c_det)

    # D2: Registry lookup-error routing.
    async def d2(p: Probe):
        reg, _, _ = two_session_registry()
        cases, lines = [
            ("hello.count_to", None),                 # valid → routes
            ("nope.count_to", "unknown server"),      # bad server
            ("count_to", "must be namespaced"),       # un-namespaced
            ("hello.bogus", "not found on server"),   # bad tool
        ], []
        ok = True
        for name, expect in cases:
            try:
                reg.lookup_tool(name)
                got = "resolved OK"
                if expect is not None:
                    ok = False
            except KeyError as e:
                got = str(e).strip('"')
                if expect is None or expect not in got:
                    ok = False
            lines.append(f"{name!r:22} → {got}")
        p.request = "Registry.lookup_tool() with valid + 3 malformed names"
        p.decision = "namespace parsed; unknown server / tool / un-namespaced each rejected"
        p.status = "PASS" if ok else "FAIL"
        p.outcome = "valid name routes; each malformed name raises a specific KeyError"
        p.commentary = ("Routing fails CLOSED and legibly: a name that can't be attributed "
                        "to a real (server, tool) is rejected with a precise reason rather "
                        "than mis-dispatched.")
        p.detail = "\n".join(lines)

    await run_probe(Probe("D2", "D. Surfaces & routing errors",
                          "Tool-name lookup rejects malformed/unknown names with reasons",
                          "negative", "deterministic"), d2)

    # E1: inferencer resolution — provider → base_url + auth.
    async def e1(p: Probe):
        from woollama import inferencers
        expect = {
            "ollama": ("http://localhost:11434/v1", None),
            "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
            "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
            "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        }
        ok, lines = True, []
        for name, (url, env) in expect.items():
            inf = inferencers.get(name)
            good = inf is not None and inf.base_url == url and inf.api_key_env == env
            ok = ok and good
            lines.append(f"{name:10} → {getattr(inf,'base_url',None)}  auth={getattr(inf,'api_key_env',None)}")
        p.request = "inferencers.get(<provider>) for the built-in clouds + ollama"
        p.decision = "each provider resolves to its OpenAI-compat base_url + key env"
        p.status = "PASS" if ok else "FAIL"
        p.outcome = "all built-in providers resolved to the documented endpoint + auth"
        p.commentary = ("This is the MODEL-axis backend resolution: the `<provider>/` prefix "
                        "selects a base_url and which env var holds the key, read at call time.")
        p.detail = "\n".join(lines)

    await run_probe(Probe("E1", "E. Inferencer resolution",
                          "Built-in providers resolve to the right endpoint + auth",
                          "config", "deterministic"), e1)

    # E2: config-file inferencer overrides/extends the built-ins.
    async def e2(p: Probe):
        from woollama import inferencers
        d = tempfile.mkdtemp(prefix="wool-inf-")
        (Path(d) / "inferencers.toml").write_text(
            '[inferencers.myvllm]\nbase_url = "http://gpu.local:8000/v1"\n'
            'api_key_env = "MYVLLM_KEY"\n\n'
            '[inferencers.ollama]\nbase_url = "http://override:9999/v1"\n')
        old = os.environ.get("WOOLLAMA_CONFIG_DIR")
        os.environ["WOOLLAMA_CONFIG_DIR"] = d
        try:
            mv = inferencers.get("myvllm")
            ov = inferencers.get("ollama")
        finally:
            if old is None:
                os.environ.pop("WOOLLAMA_CONFIG_DIR", None)
            else:
                os.environ["WOOLLAMA_CONFIG_DIR"] = old
        added = mv is not None and mv.base_url == "http://gpu.local:8000/v1"
        overrode = ov is not None and ov.base_url == "http://override:9999/v1"
        p.request = "inferencers.toml adds 'myvllm' and overrides built-in 'ollama' base_url"
        p.decision = f"myvllm={getattr(mv,'base_url',None)}; ollama={getattr(ov,'base_url',None)}"
        p.status = "PASS" if (added and overrode) else "FAIL"
        p.outcome = "config-file inferencers are added AND override built-ins by name"
        p.commentary = ("Routing targets are extensible without code: any OpenAI-compat "
                        "backend can be added, and a built-in can be repointed, via config.")
        p.detail = (f"myvllm.base_url={getattr(mv,'base_url',None)}\n"
                    f"ollama.base_url={getattr(ov,'base_url',None)} (built-in default is :11434)")

    await run_probe(Probe("E2", "E. Inferencer resolution",
                          "Config-file inferencers add new backends and override built-ins",
                          "config", "deterministic"), e2)


# ===========================================================================
# LIVE probes (against a booted woollama over real Ollama)
# ===========================================================================

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def live_probes(base: str, sock: str, ollama_ok: bool, have_model: bool) -> None:
    print("\n── LIVE (booted woollama; real Ollama)" if ollama_ok
          else "\n── LIVE (Ollama unreachable — model probes will SKIP)")

    async def post(model, messages, **extra):
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(f"{base}/v1/chat/completions",
                             json={"model": model, "messages": messages, **extra})
            return r

    # A3: unknown recipe → 404
    async def a3(p: Probe):
        r = await post("woollama/does-not-exist", [{"role": "user", "content": "hi"}])
        body = r.json()
        p.request = "POST /v1/chat/completions model=woollama/does-not-exist"
        p.decision = f"HTTP {r.status_code}; error.type={body.get('error',{}).get('type')}"
        p.status = "PASS" if r.status_code == 404 else "FAIL"
        p.outcome = "unknown recipe rejected at the woollama/ branch"
        p.commentary = "Distinct route from an unknown *namespace* (A4) — here the namespace is known."
        p.detail = json.dumps(body, indent=2)
    await run_probe(Probe("A3", "A. Model-namespace routing",
                          "woollama/<unknown recipe> → 404", "negative", "live"), a3)

    # A4: unknown namespace → 400, distinct message
    async def a4(p: Probe):
        r = await post("totally-bogus/x", [{"role": "user", "content": "hi"}])
        body = r.json()
        msg = body.get("error", {}).get("message", "")
        p.request = "POST model=totally-bogus/x  (neither woollama/ nor a known inferencer)"
        p.decision = f"HTTP {r.status_code}: {msg}"
        p.status = "PASS" if (r.status_code == 400 and "unknown model namespace" in msg) else "FAIL"
        p.outcome = "unknown namespace rejected with the namespace-guidance message"
        p.commentary = ("Same 400 as A5/A7 but a DIFFERENT route+message — this is the "
                        "fall-through when neither woollama/ nor a known provider matches.")
        p.detail = json.dumps(body, indent=2)
    await run_probe(Probe("A4", "A. Model-namespace routing",
                          "Unknown namespace → 400 (distinct message)", "negative", "live"), a4)

    # A5: known cloud provider, missing key → 400 via passthrough headers()
    async def a5(p: Probe):
        r = await post("anthropic/claude-haiku-4-5", [{"role": "user", "content": "hi"}])
        body = r.json()
        msg = body.get("error", {}).get("message", "")
        keyless = not os.environ.get("ANTHROPIC_API_KEY")
        p.request = "POST model=anthropic/claude-haiku-4-5  (no ANTHROPIC_API_KEY set)"
        p.decision = f"HTTP {r.status_code}: {msg}"
        if not keyless:
            p.status, p.outcome = "SKIP", "ANTHROPIC_API_KEY is set — can't show the keyless path"
            p.mode = "skip"
        else:
            p.status = "PASS" if (r.status_code == 400 and "ANTHROPIC_API_KEY" in msg) else "FAIL"
            p.outcome = "passthrough route resolves the provider, then fails fast on the missing key"
        p.commentary = ("PASSTHROUGH path (no orchestration): the provider IS known, so routing "
                        "succeeds; the 400 is the credential check in inferencer.headers(). "
                        "Contrast A7 (same failure, orchestration path).")
        p.detail = json.dumps(body, indent=2)
    await run_probe(Probe("A5", "A. Model-namespace routing",
                          "Known cloud provider, missing key → 400 (passthrough)", "negative", "live"), a5)

    # A6: recipe → unsupported inferencer → 501
    async def a6(p: Probe):
        r = await post("woollama/route-bogus", [{"role": "user", "content": "hi"}])
        body = r.json()
        p.request = "POST model=woollama/route-bogus  (recipe.inferencer = no-such-provider/m)"
        p.decision = f"HTTP {r.status_code}; type={body.get('error',{}).get('type')}"
        p.status = "PASS" if r.status_code == 501 else "FAIL"
        p.outcome = "recipe resolves, but its inferencer is unknown → 501 not_implemented"
        p.commentary = "Routing reaches orchestration, then rejects an unroutable backend."
        p.detail = json.dumps(body, indent=2)
    await run_probe(Probe("A6", "A. Model-namespace routing",
                          "Recipe → unsupported inferencer → 501", "negative", "live"), a6)

    # A7: recipe → cloud inferencer, missing key → 400 via orchestrate headers()
    async def a7(p: Probe):
        r = await post("woollama/route-cloud", [{"role": "user", "content": "hi"}])
        body = r.json()
        msg = body.get("error", {}).get("message", "")
        keyless = not os.environ.get("ANTHROPIC_API_KEY")
        p.request = "POST model=woollama/route-cloud  (recipe.inferencer = anthropic/…, no key)"
        p.decision = f"HTTP {r.status_code}: {msg}"
        if not keyless:
            p.status, p.outcome, p.mode = "SKIP", "ANTHROPIC_API_KEY set — keyless path unavailable", "skip"
        else:
            p.status = "PASS" if (r.status_code == 400 and "ANTHROPIC_API_KEY" in msg) else "FAIL"
            p.outcome = "orchestration route fails fast on the missing key (same as A5, other path)"
        p.commentary = ("ORCHESTRATION path: the cloud round-trip is the one routing+backend "
                        "combo we cannot exercise live without a key — this is its honest SKIP/"
                        "negative. Auth+routing are unit-tested on the emit side.")
        p.detail = json.dumps(body, indent=2)
    await run_probe(Probe("A7", "A. Model-namespace routing",
                          "Recipe → cloud inferencer, missing key → 400 (orchestration)", "negative", "live"), a7)

    # claude-code backend route — represented as a gated SKIP (3rd backend path)
    cc = Probe("A8", "A. Model-namespace routing",
               "Recipe → claude-code backend (subprocess route)", "integration", "skip",
               status="SKIP",
               request="model=woollama/cc-assistant (inferencer = claude-code/haiku)",
               decision="would route to the local `claude` CLI subprocess (keyless)",
               outcome="SKIPPED — gated: needs the `claude` CLI on PATH and costs real money",
               commentary=("Third backend route alongside ollama (live) and cloud (A7). Routing "
                           "selection is unit-tested; the live subprocess run is opt-in "
                           "(WOOLLAMA_TEST_CLAUDE_CODE=1)."))
    emit(cc)

    # F1: same route over UDS and TCP
    async def f1(p: Probe):
        tcp = httpx.get(f"{base}/v1/tools", timeout=5).json()["tools"]
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=sock)) as uc:
            uds = (await uc.get("http://localhost/v1/tools", timeout=5)).json()["tools"]
        p.request = "GET /v1/tools over the TCP port AND over the Unix socket"
        p.decision = "both transports hit the same app + shared registry"
        p.status = "PASS" if (tcp == uds and tcp) else "FAIL"
        p.outcome = f"identical tool list on both transports ({len(tcp)} tools)"
        p.commentary = "Transport routing: one app, two listeners, one registry."
        p.detail = f"TCP: {tcp}\nUDS: {uds}"
    await run_probe(Probe("F1", "F. Transport routing",
                          "Same routes answer over UDS and TCP", "integration", "live"), f1)

    if not (ollama_ok and have_model):
        for pid, intent in [("A1", "ollama/<model> → passthrough"),
                            ("A2", "woollama/<recipe> → orchestration"),
                            ("B1", "LIVE cross-session fan-out (MCP progress)")]:
            emit(Probe(pid, "A/B. live model routing", intent, "integration", "skip",
                       status="SKIP",
                       outcome=f"SKIPPED — {'Ollama unreachable' if not ollama_ok else MODEL+' not pulled'}",
                       commentary="Deterministic counterparts (B-det) still prove the routing logic."))
        return

    # A1: ollama/<model> → passthrough (no orchestration)
    async def a1(p: Probe):
        r = await post(f"ollama/{MODEL}", [{"role": "user", "content": "Reply with exactly: pong"}])
        body = r.json()
        msg = body["choices"][0]["message"]
        p.request = f"POST model=ollama/{MODEL}  (bare pass-through)"
        p.decision = "prefix stripped → forwarded to Ollama's /v1/chat/completions, no tool loop"
        p.status = "PASS" if (r.status_code == 200 and msg.get("content")) else "FAIL"
        p.outcome = f"passthrough returned content: {msg.get('content','').strip()[:60]!r}"
        p.commentary = "Integration: proves the passthrough route reaches Ollama and returns its answer."
        p.detail = f"finish_reason={body['choices'][0].get('finish_reason')}; tool_calls={msg.get('tool_calls')}"
    await run_probe(Probe("A1", "A. Model-namespace routing",
                          "ollama/<model> → passthrough to Ollama", "integration", "live"), a1)

    # A2: woollama/streamer → orchestration (loop hidden)
    async def a2(p: Probe):
        r = await post("woollama/streamer", [{"role": "user", "content": "Count to 3."}])
        body = r.json()
        msg = body["choices"][0]["message"]
        p.request = "POST model=woollama/streamer  (recipe orchestration)"
        p.decision = "routed to orchestrate(); hello.count_to dispatched internally"
        p.status = "PASS" if (r.status_code == 200 and msg.get("content")
                              and not msg.get("tool_calls")) else "FAIL"
        p.outcome = f"final answer only, no tool_calls leaked: {msg.get('content','').strip()[:60]!r}"
        p.commentary = ("Integration of the orchestration route. NOTE: HTTP hides tool results, "
                        "so this can't prove WHICH session was hit — B1 does that, live, via "
                        "MCP progress, and B-det proves it definitively.")
        p.detail = f"tool_calls={msg.get('tool_calls')}"
    await run_probe(Probe("A2", "A. Model-namespace routing",
                          "woollama/<recipe> → orchestration (loop hidden)", "integration", "live"), a2)

    # B1: LIVE cross-session fan-out, observed via MCP progress notifications, ×3
    async def b1(p: Probe):
        from fastmcp import Client

        def make_handler(sink):
            async def handler(params):
                d = params.data
                sink.append(d["msg"] if isinstance(d, dict) and "msg" in d else str(d))
            return handler

        runs, both_count, run_logs = 3, 0, []
        for i in range(runs):
            logs: list[str] = []
            async with Client(f"{base}/mcp", log_handler=make_handler(logs)) as mc:
                await mc.call_tool("chat", {
                    "recipe": "textcounter",
                    "messages": [{"role": "user", "content": "Count the words in 'a b c d', then count to that."}],
                })
            hit_textops = any("→ textops.word_count" in m for m in logs)
            hit_hello = any("→ hello.count_to" in m for m in logs)
            if hit_textops and hit_hello:
                both_count += 1
            run_logs.append(f"run {i+1}: textops={hit_textops} hello={hit_hello}  "
                            f"calls={[m for m in logs if m.startswith('→')]}")
        p.request = ("MCP chat tool, recipe=textcounter (allow-list spans textops+hello), "
                     f"run {runs}×; routing observed via the chat tool's progress notifications")
        p.decision = f"{both_count}/{runs} runs fanned out to BOTH sessions"
        if both_count == runs:
            p.status = "PASS"
        elif both_count == 0:
            p.status = "FAIL"
        else:
            p.status = "PARTIAL"
        p.outcome = (f"{both_count}/{runs} runs routed to both textops AND hello sessions "
                     "(observed live via streaming-3 progress)")
        p.commentary = ("LIVE attribution — the progress notifications reveal which session each "
                        "tool_call hit, so this proves real cross-session routing end-to-end. "
                        "Model-DEPENDENT: qwen3:14b decides whether to call both tools, so a "
                        "ratio < 3/3 reflects the MODEL's choices, not a routing fault — B-det "
                        "proves the routing itself is correct whenever a call is made.")
        p.detail = "\n".join(run_logs)
    await run_probe(Probe("B1", "B. Tool routing across sessions",
                          "LIVE cross-session fan-out, proven via MCP progress (×3)",
                          "attribution", "live"), b1)


# ===========================================================================
# HTML report
# ===========================================================================

def render_html(meta: dict) -> str:
    badge = {"PASS": "#1a7f37", "FAIL": "#cf222e", "PARTIAL": "#bf8700", "SKIP": "#6e7781"}
    proves_desc = {
        "attribution": "proves the RIGHT call reached the RIGHT session/backend",
        "integration": "proves the route works end-to-end (not which session)",
        "negative": "proves a bad/forbidden route is rejected, legibly",
        "config": "proves backend resolution from built-ins + config",
    }
    counts = {s: sum(1 for p in probes if p.status == s) for s in badge}
    findings = [p for p in probes if p.status in ("FAIL", "PARTIAL", "SKIP")]

    def esc(s): return html.escape(str(s))

    rows = []
    axes: dict[str, list[Probe]] = {}
    for p in probes:
        axes.setdefault(p.axis, []).append(p)
    for axis, ps in axes.items():
        rows.append(f'<h2>{esc(axis)}</h2>')
        for p in ps:
            rows.append(f'''
<details class="probe" data-status="{p.status}" data-mode="{p.mode}" open>
  <summary>
    <span class="b" style="background:{badge[p.status]}">{p.status}</span>
    <span class="pid">{esc(p.id)}</span>
    <span class="intent">{esc(p.intent)}</span>
    <span class="tags">{esc(p.mode)} · {esc(p.proves)}</span>
  </summary>
  <div class="body">
    <div class="kv"><b>Proves</b><span>{esc(proves_desc.get(p.proves, p.proves))}</span></div>
    <div class="kv"><b>Request</b><span>{esc(p.request)}</span></div>
    <div class="kv"><b>Routing decision</b><span>{esc(p.decision)}</span></div>
    <div class="kv"><b>Outcome</b><span>{esc(p.outcome)}</span></div>
    <div class="kv"><b>Commentary</b><span>{esc(p.commentary)}</span></div>
    {"<pre>"+esc(p.detail)+"</pre>" if p.detail else ""}
  </div>
</details>''')

    findings_html = "".join(
        f'<li><span class="b" style="background:{badge[p.status]}">{p.status}</span> '
        f'<b>{esc(p.id)}</b> — {esc(p.outcome or p.intent)}</li>' for p in findings
    ) or "<li>None — every probe passed.</li>"

    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>woollama routing validation</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1f2328}}
 h1{{margin-bottom:.2rem}} h2{{margin-top:2rem;border-bottom:2px solid #d0d7de;padding-bottom:.3rem}}
 .sub{{color:#6e7781}}
 .summary{{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0}}
 .pill{{padding:.3rem .7rem;border-radius:999px;color:#fff;font-weight:600}}
 .findings{{background:#fff8c5;border:1px solid #d4a72c;border-radius:8px;padding:.8rem 1rem;margin:1rem 0}}
 .findings ul{{margin:.4rem 0 0;padding-left:1rem}} .findings li{{margin:.25rem 0}}
 .bar{{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #d0d7de;z-index:1}}
 .bar button{{font:inherit;padding:.3rem .7rem;margin-right:.4rem;border:1px solid #d0d7de;border-radius:6px;background:#f6f8fa;cursor:pointer}}
 .bar button.on{{background:#0969da;color:#fff;border-color:#0969da}}
 details.probe{{border:1px solid #d0d7de;border-radius:8px;margin:.5rem 0;background:#fff}}
 summary{{cursor:pointer;padding:.6rem .8rem;display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}}
 .b{{color:#fff;font-weight:700;font-size:11px;padding:.15rem .5rem;border-radius:5px}}
 .pid{{font-family:ui-monospace,monospace;color:#6e7781}}
 .intent{{flex:1;min-width:240px}} .tags{{font-size:12px;color:#6e7781;font-family:ui-monospace,monospace}}
 .body{{padding:.2rem .9rem .9rem;border-top:1px solid #eaeef2}}
 .kv{{display:flex;gap:.8rem;padding:.25rem 0}} .kv b{{flex:0 0 130px;color:#57606a}} .kv span{{flex:1}}
 pre{{background:#f6f8fa;border:1px solid #eaeef2;border-radius:6px;padding:.6rem;overflow:auto;font-size:12.5px;white-space:pre-wrap}}
 .legend{{color:#6e7781;font-size:13px;margin-top:2rem;border-top:1px solid #d0d7de;padding-top:1rem}}
</style></head><body>
<h1>woollama — routing validation</h1>
<div class="sub">{esc(meta['when'])} · Ollama: {esc(meta['ollama'])} · model: {esc(meta['model'])} · cloud keys: {esc(meta['keys'])}</div>

<div class="summary">
 {"".join(f'<span class="pill" style="background:{badge[s]}">{counts[s]} {s}</span>' for s in badge if counts[s])}
</div>

<div class="findings"><b>Findings (failures, partials &amp; skips — read these first):</b>
 <ul>{findings_html}</ul>
</div>

<div class="bar">
 <button class="on" onclick="flt(this,'all')">All</button>
 <button onclick="flt(this,'issues')">Failures &amp; partials</button>
 <button onclick="flt(this,'live')">Live only</button>
 <button onclick="flt(this,'deterministic')">Deterministic only</button>
</div>

{"".join(rows)}

<div class="legend">
 <b>How to read this.</b> Each probe states what it <i>proves</i>:
 <b>attribution</b> ({proves_desc['attribution']}), <b>integration</b> ({proves_desc['integration']}),
 <b>negative</b> ({proves_desc['negative']}), <b>config</b> ({proves_desc['config']}).
 <b>Deterministic</b> probes use a scripted inferencer + recording sessions (no model) for
 definitive proof; <b>live</b> probes hit a booted woollama over real Ollama. Live model
 fan-out (B1) is model-dependent and reported as a ratio — a ratio &lt; 3/3 is the model's
 choice, not a routing fault (B-det proves the routing). Re-run:
 <code>uv run --extra dev python examples/routing_validation.py</code>
</div>

<script>
 function flt(btn, mode){{
   document.querySelectorAll('.bar button').forEach(b=>b.classList.remove('on'));
   btn.classList.add('on');
   document.querySelectorAll('details.probe').forEach(d=>{{
     const st=d.dataset.status, md=d.dataset.mode;
     let show=true;
     if(mode==='issues') show=(st==='FAIL'||st==='PARTIAL');
     else if(mode==='live') show=(md==='live');
     else if(mode==='deterministic') show=(md==='deterministic');
     d.style.display=show?'':'none';
   }});
 }}
</script>
</body></html>'''


# ===========================================================================
# Driver
# ===========================================================================

async def main() -> int:
    print("woollama ROUTING validation\n" + "=" * 50)

    await deterministic_probes()

    # Boot a live server with a recipe set that covers every backend route.
    rt = tempfile.mkdtemp(prefix="wool-routing-")
    recipes_toml = (REPO / "src/woollama/defaults/recipes.toml").read_text()
    recipes_toml += (
        '\n[recipes.route-cloud]\ninferencer = "anthropic/claude-haiku-4-5"\n'
        'tools = []\nsystem = "x"\n'
        '\n[recipes.route-bogus]\ninferencer = "no-such-provider/m"\n'
        'tools = []\nsystem = "x"\n')
    (Path(rt) / "recipes.toml").write_text(recipes_toml)
    port = free_port()
    sock = os.path.join(rt, "woollama.sock")
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "WOOLLAMA_ADDRESS": f"127.0.0.1:{port}",
           "WOOLLAMA_CONFIG_DIR": rt, "XDG_RUNTIME_DIR": rt}
    print(f"\nbooting woollama on {base}")
    proc = subprocess.Popen([sys.executable, "-m", "woollama"], cwd=REPO, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ollama_ok = have_model = False
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{base}/v1/models", timeout=0.5).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        try:
            tags = httpx.get("http://localhost:11434/api/tags", timeout=2)
            ollama_ok = tags.status_code == 200
        except Exception:
            ollama_ok = False
        if ollama_ok:
            ids = [m["id"] for m in httpx.get(f"{base}/v1/models", timeout=5).json()["data"]]
            have_model = any(MODEL in i for i in ids)
        await live_probes(base, sock, ollama_ok, have_model)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    keys = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY") if os.environ.get(k)]
    meta = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ollama": "reachable" if ollama_ok else "unreachable",
            "model": MODEL if have_model else "(not available)",
            "keys": ", ".join(keys) if keys else "none set"}
    OUT_HTML.write_text(render_html(meta))
    OUT_JSON.write_text(json.dumps([asdict(p) for p in probes], indent=2))

    print("\n" + "=" * 50)
    for s in ("PASS", "PARTIAL", "FAIL", "SKIP"):
        n = sum(1 for p in probes if p.status == s)
        if n:
            print(f"  {s:8} {n}")
    print(f"\nreport: {OUT_HTML}\njson:   {OUT_JSON}")
    return 1 if any(p.status == "FAIL" for p in probes) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
