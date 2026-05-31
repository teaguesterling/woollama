# mcp-textops

A second tiny FastMCP example server alongside `mcp-hello`. Exists so
woollama's multi-server discovery + namespacing has something real to
demonstrate.

Three trivial text-manipulation tools:

- `uppercase(text)` → `text.upper()`
- `word_count(text)` → number of whitespace-separated words
- `reverse_text(text)` → reversed characters

When woollama connects to both `mcp-hello` and `mcp-textops`, the unified
tool registry exposes them as:

```
hello.hello, hello.count_to, hello.ask_user
textops.uppercase, textops.word_count, textops.reverse_text
```

Recipes reference tools by these namespaced names.

## Running standalone

```sh
python server.py    # stdio transport; for direct probing
```
