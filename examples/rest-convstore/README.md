# rest-convstore

A reference **REST** conversation-store server — a second implementation of
woollama's conversation-store contract alongside [`mcp-convstore`](../mcp-convstore),
proving the `ConversationStoreProvider` seam is transport-agnostic (MCP *and*
plain HTTP, same four operations). This one is **file-backed**, so transcripts
**persist across restarts** (the in-memory MCP example does not).

woollama is a *client* to it (`conversations.HttpStoreProvider`); the bytes live
here, one JSON file per thread under `$CONVSTORE_DIR`. woollama holds only the
opaque thread id — it never owns the transcript.

## REST surface

The `ConversationStoreProvider` contract (`create` / `get` / `append` / `delete`):

| Method | Path | Effect | Response |
|---|---|---|---|
| `PUT` | `/threads/{id}` | create an empty thread (idempotent) | `204` |
| `GET` | `/threads/{id}` | the message list (`[]` if absent) | `200` JSON array |
| `PATCH` | `/threads/{id}` | append messages (JSON-array body) | `200 {"count"}` |
| `DELETE` | `/threads/{id}` | delete the thread | `204` |

The **provider** mints the thread id (a UUID) and `PUT`s it, so this server
needs no id-minting logic and create is idempotent.

## Wiring it into woollama

Run the store, then point woollama at it with the `conversationStore` key in
`mcp.json` (the typed `http` form):

```sh
CONVSTORE_DIR=/var/lib/woollama/threads python server.py --port 9000
```

```json
{
  "conversationStore": { "type": "http", "url": "http://127.0.0.1:9000" },
  "mcpServers": { }
}
```

Once wired, **every** non-claude model (`ollama/*`, cloud providers, and
`woollama/<recipe>`) becomes stateful on `/v1/responses` + `/v1/conversations`.
See [Configuration](../../docs/configuration.md) and the
[Conversations design](../../docs/conversations-api-design.md) §10.

| Env / flag | Default | Effect |
|---|---|---|
| `CONVSTORE_DIR` | `$TMPDIR/woollama-rest-convstore` | Where the per-thread JSON files are stored. |
| `--port` / `CONVSTORE_PORT` | `9000` | Listen port. |
| `--host` | `127.0.0.1` | Bind host. |

Thread ids are constrained to `[A-Za-z0-9_-]+` so a crafted id can't escape the
store directory.
