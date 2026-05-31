# cosmic-mcp-hello — reference + smoke-test MCP server

A tiny FastMCP-based MCP server that demonstrates the three load-bearing
behaviors cos-fab depends on. Lives here as:

- **A smoke test** for the cos-fab MCP integration. Run the probe client; if
  all three tests pass, your install can speak MCP correctly.
- **A reference implementation** for users (and us) writing wrappers in
  cos-fab's conventions — the same shape as `cosmic-mcp-fabric` and
  `cosmic-mcp-ollama`.
- **A controllable test fixture** as we build out the daemon's MCP
  integration — no fabric, no Ollama, no flaky network.

## The three load-bearing things

| tool | exercises | mcp.json `features.streaming` value implied |
|---|---|---|
| `hello(name)` | basic tool round-trip | — |
| `count_to(n)` | progress notifications mid-call (the **streaming convention**) | `progress-typed-events` |
| `ask_user(question)` | server → client callback via **elicitation** (the bidirectional case) | — |

## Running

```sh
# Direct run (stdio):
python server.py

# Probe it end-to-end:
uv run --with mcp --with fastmcp python probe_client.py
```

`probe_client.py` prints the result of each test and a verdict line. Expected
on a healthy install:

```
T1 (basic tool):           round-trip OK
T2 (progress streaming):   OK
T3 (bidirectional/elicit): OK
```

## In your mcp.json (when used as a real server)

```jsonc
{
  "mcpServers": {
    "hello": {
      "command": "python",
      "args": ["/path/to/examples/mcp-hello/server.py"],
      "features": {
        "streaming": "progress-typed-events"
      }
    }
  }
}
```

## What this isn't

It is not part of the production cos-fab bundle. It ships as documentation +
test fixture; users don't connect to it for real work. If we later want a
diagnostic / health-check tool that's pip-installable, this graduates to a
separate `cosmic-mcp-hello` repo. Today it lives next to the design docs.

## Design notes

- **Capabilities advertised** (default-on with FastMCP): `prompts`,
  `resources`, `tools`, `logging`, plus the
  `io.modelcontextprotocol/ui` extension. We use only `tools` here; the
  presence of the others is a free win for anyone building a more
  elaborate server.
- **Elicitation vs sampling**: this example uses `elicitation` for the
  bidirectional test because it's the simplest standard primitive. The
  TL.3 case (handler invokes router tools) in cos-fab's architecture is
  cleaner via *two MCP connections* — see
  [`doc/design/tool-calling-plan.md`](../../doc/design/tool-calling-plan.md)
  and the conversation history.
- **Why FastMCP**: it gives us prompts/resources/tools/progress out of the
  box. We standardize on it for our own MCP servers.
