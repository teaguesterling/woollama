# mcp-convstore

A reference MCP **conversation-store** server — the external owner of transcript
bytes that makes woollama's non-claude models stateful (issue #2), without
woollama ever holding the bytes itself. This is the load-bearing example for
woollama's core principle: *woollama routes conversation handles; backends own
the state.*

Four tools form the `ConversationStoreProvider` contract:

- `create_thread()` → a new thread id (the store mints it)
- `get_thread(thread_id)` → the message list (`[]` for a fresh thread)
- `append_turn(thread_id, messages)` → `{"ok", "count"}`
- `delete_thread(thread_id)` → `{"ok": true}`

The transcript lives in **this** process (a plain in-memory dict here; a real
provider would back it with sqlite/postgres/a file). woollama holds only the
opaque `thread_id` and asks this server for the history each turn. Every tool
returns an explicit JSON string (`json.dumps(...)`) so the wire shape is
unambiguous for woollama's `McpStoreProvider` to parse.

## Wiring it into woollama

Register it in `mcp.json` and name it with `WOOLLAMA_CONVSTORE_SERVER`:

```json
{
  "mcpServers": {
    "convstore": {
      "command": "python",
      "args": ["${WOOLLAMA_EXAMPLES_DIR}/mcp-convstore/server.py"]
    }
  }
}
```

```sh
WOOLLAMA_CONVSTORE_SERVER=convstore woollama
```

Once wired, **every** non-claude model (`ollama/*`, cloud providers, and
`woollama/<recipe>`) becomes stateful on `/v1/responses` + `/v1/conversations`.
See [Configuration](../../docs/configuration.md) and the
[Conversations design](../../docs/conversations-api-design.md) §10.

> ⚠️ The in-memory dict is **not** persistent — it's for demonstration and tests.
> Threads vanish when the server stops. Back it with real storage for anything
> beyond a demo.

## Running standalone

```sh
python server.py    # stdio transport; for direct probing
```
