"""Hermetic tests for the `woollama` CLI entrypoint dispatch (`__main__.main`).

The real server / stdio serving is exercised by test_integration.py (which spawns
`python -m woollama` as a subprocess — unmeasured by coverage). Here we mock the
heavy seams (binding, uvicorn, serve) and assert `main()` routes each argv to the
right mode. This is pure arg-routing logic that no other hermetic test touches.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from woollama import __main__, __version__


def test_main_version_flag_prints_and_returns_0(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["woollama", "--version"])
    rc = __main__.main()
    assert rc == 0
    assert __version__ in capsys.readouterr().out


def test_main_mcp_subcommand_dispatches_to_serve(monkeypatch):
    """`woollama mcp` runs the stdio MCP server (serve), not the HTTP server."""
    monkeypatch.setattr(sys, "argv", ["woollama", "mcp"])
    import woollama.mcp_server as mcp_server
    called = {}
    monkeypatch.setattr(mcp_server, "serve", lambda: called.setdefault("served", True))
    # If it wrongly took the HTTP path it would touch binding — make that explode.
    monkeypatch.setattr(__main__.binding, "open_sockets",
                        lambda: (_ for _ in ()).throw(AssertionError("took HTTP path")))
    rc = __main__.main()
    assert rc == 0 and called.get("served") is True


def test_main_default_starts_http_server_on_bound_sockets(monkeypatch):
    """The default (no-arg) path binds sockets, serves them via uvicorn, and
    cleans up in `finally` — all mocked so nothing real binds or serves."""
    monkeypatch.setattr(sys, "argv", ["woollama"])
    fake_listeners = SimpleNamespace(
        sockets=["SOCK"], sock_path="/run/woollama.sock",
        tcp_host="127.0.0.1", tcp_port=12345)
    monkeypatch.setattr(__main__.binding, "open_sockets", lambda: fake_listeners)
    cleaned = {}
    monkeypatch.setattr(__main__.binding, "cleanup", lambda lst: cleaned.setdefault("c", lst))
    monkeypatch.setattr(__main__.binding, "addr_file_path", lambda: "/run/woollama.addr")

    run_args = {}

    class _Server:
        def __init__(self, config):
            pass

        def run(self, sockets=None):
            run_args["sockets"] = sockets

    monkeypatch.setattr(__main__.uvicorn, "Server", _Server)
    monkeypatch.setattr(__main__.uvicorn, "Config", lambda app, **kw: ("cfg", app))

    rc = __main__.main()
    assert rc == 0
    assert run_args["sockets"] == ["SOCK"]       # served the pre-bound sockets
    assert cleaned["c"] is fake_listeners         # cleanup ran in finally
