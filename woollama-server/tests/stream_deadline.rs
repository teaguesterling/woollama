//! A streamed answer must not be cut by a request deadline, and a stream that DOES fail must say so.
//!
//! Measured 2026-11-22: 36 of 37 pi failures "Stream ended without finish_reason" were woollamad's
//! own 180 s upstream deadline. `reqwest`'s `timeout` is a TOTAL deadline that includes reading the
//! body, so a long generation was cut at 180 s after dispatch. The relay then just ended the body:
//! no SSE error event, no final chunk, no `[DONE]`. The access line, written at header time, said
//! `200` in about 100 ms. Directly against the device, the same kind of stream ran 651 s and ended
//! normally.
//!
//! These drive the REAL binary, with stderr in a file, because the end-of-stream log line is part
//! of the contract and an in-process test cannot see `eprintln!`. `WOOLLAMA_STREAM_IDLE_TIMEOUT`
//! scales the idle timeout down to one second, so "longer than the timeout" takes seconds.
//!
//! Each scenario runs through all three streaming relays:
//!   device/<m>   a pooled (management_url) inferencer — `passthrough_pooled`, the path pi uses
//!   plain/<m>    an inferencer with no pool — `passthrough_stream`
//!   ollama/<m>   with `num_ctx` — `native_stream` (NDJSON translated to SSE)

use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::extract::Path as AxPath;
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde_json::{json, Value};

/// The idle timeout the daemon runs with in these tests.
const IDLE_SECS: f64 = 1.0;

/// The models the fake device reports as running. The model id selects the behaviour.
const MODELS: [&str; 4] = ["slow", "silent", "drop", "endless"];

// --- the fake upstream ----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// OpenAI SSE (`/chat/completions`).
    Sse,
    /// ollama native NDJSON (`/api/chat`).
    Ndjson,
}

fn frame(shape: Shape, text: &str) -> String {
    match shape {
        Shape::Sse => format!(
            "data: {}\n\n",
            json!({"id": "x", "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]})
        ),
        Shape::Ndjson => format!("{}\n", json!({"message": {"role": "assistant", "content": text}, "done": false})),
    }
}

fn finish(shape: Shape) -> String {
    match shape {
        Shape::Sse => format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"id": "x", "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        ),
        Shape::Ndjson => format!("{}\n", json!({"message": {"content": ""}, "done": true, "done_reason": "stop"})),
    }
}

/// Half a frame: what is on the wire when a connection dies in the middle of one.
fn partial(shape: Shape) -> &'static str {
    match shape {
        Shape::Sse => "data: {\"id\":\"x\",\"choi",
        Shape::Ndjson => "{\"message\":{\"rol",
    }
}

fn stream_body(model: &str, shape: Shape) -> Response {
    let model = model.to_string();
    let body = async_stream::stream! {
        match model.as_str() {
            // 12 frames, 250 ms apart: 3 s in all, three times the idle timeout, and never idle
            // for longer than a quarter of it.
            "slow" => {
                for i in 0..12 {
                    yield Ok::<Bytes, std::io::Error>(Bytes::from(frame(shape, &format!("t{i} "))));
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                yield Ok(Bytes::from(finish(shape)));
            }
            // One frame, then nothing for ten idle timeouts.
            "silent" => {
                yield Ok(Bytes::from(frame(shape, "t0 ")));
                tokio::time::sleep(Duration::from_secs_f64(IDLE_SECS * 10.0)).await;
                yield Ok(Bytes::from(finish(shape)));
            }
            // One frame and half of the next, then the connection dies. hyper aborts a chunked
            // body whose stream yields an error, so the client sees a truncated body.
            "drop" => {
                yield Ok(Bytes::from(frame(shape, "t0 ")));
                tokio::time::sleep(Duration::from_millis(100)).await;
                yield Ok(Bytes::from(partial(shape)));
                tokio::time::sleep(Duration::from_millis(100)).await;
                yield Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "fake upstream reset"));
            }
            // Frames every 100 ms for a minute: for the client that goes away.
            _ => {
                for i in 0..600 {
                    yield Ok(Bytes::from(frame(shape, &format!("t{i} "))));
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                yield Ok(Bytes::from(finish(shape)));
            }
        }
    };
    let ctype = if shape == Shape::Sse { "text/event-stream" } else { "application/x-ndjson" };
    Response::builder().header("content-type", ctype).body(Body::from_stream(body)).unwrap()
}

async fn chat(Json(b): Json<Value>) -> Response {
    stream_body(b["model"].as_str().unwrap_or(""), Shape::Sse)
}

async fn native_chat(Json(b): Json<Value>) -> Response {
    stream_body(b["model"].as_str().unwrap_or(""), Shape::Ndjson)
}

/// tiiny_os 0.1.38 `running` shape (see `pool_device_404.rs`), with every test model resident.
async fn running(AxPath(rest): AxPath<String>) -> Json<Value> {
    let _ = rest;
    let instances: Vec<Value> = MODELS.iter().map(|id| json!({"model_id": id, "capabilities": []})).collect();
    Json(json!({"object": "list", "running": MODELS, "pending": [],
                "instances": {"running": instances, "pending": []}}))
}

async fn start(AxPath(_rest): AxPath<String>) -> Json<Value> {
    Json(json!({"message": "start loading", "progress": 0}))
}

async fn spawn_upstream() -> String {
    let router = Router::new()
        .route("/api/v1/models/{*rest}", get(running).post(start))
        .route("/chat/completions", post(chat))
        .route("/api/chat", post(native_chat));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

// --- the daemon -----------------------------------------------------------------------------------

struct Daemon {
    child: Child,
    base: String,
    log: PathBuf,
    _dir: tempfile::TempDir,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Daemon {
    fn log(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }

    /// The `stream-end` line for `model`, waiting briefly: the line is written when the relay
    /// finishes, which can be just after the client has read the last byte.
    async fn stream_end_line(&self, model: &str) -> Option<String> {
        let needle = format!("stream-end model={model} ");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(l) = self.log().lines().find(|l| l.contains(&needle)) {
                return Some(l.to_string());
            }
            if Instant::now() > deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

// The child is reaped by `Daemon::drop` (kill + wait); clippy can't see through the struct.
#[allow(clippy::zombie_processes)]
async fn spawn_daemon() -> Daemon {
    let upstream = spawn_upstream().await;
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("cfg");
    let run = dir.path().join("run");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(cfg.join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::fs::write(
        cfg.join("inferencers.toml"),
        format!(
            "[inferencers.device]\nbase_url=\"{u}\"\nmanagement_url=\"{u}\"\nmanagement_protocol=\"device\"\n\n\
             [inferencers.plain]\nbase_url=\"{u}\"\n",
            u = upstream
        ),
    )
    .unwrap();

    let log = dir.path().join("daemon.log");
    let child = Command::new(env!("CARGO_BIN_EXE_woollamad"))
        .env("WOOLLAMA_CONFIG_DIR", &cfg)
        .env("XDG_RUNTIME_DIR", &run)
        .env("WOOLLAMA_STATE_DIR", &run)
        .env("WOOLLAMA_EXAMPLES_DIR", "/nonexistent-for-test")
        .env("WOOLLAMA_OLLAMA_URL", &upstream)
        .env("WOOLLAMA_STREAM_IDLE_TIMEOUT", IDLE_SECS.to_string())
        .env_remove("WOOLLAMA_TOKEN")
        .env_remove("WOOLLAMA_ADDRESS")
        .stdout(std::process::Stdio::null())
        .stderr(std::fs::File::create(&log).unwrap())
        .spawn()
        .expect("spawn woollamad");

    let addr_file = run.join("woollama.addr");
    for _ in 0..200 {
        if let Ok(s) = std::fs::read_to_string(&addr_file) {
            let s = s.trim();
            if !s.is_empty() {
                return Daemon { child, base: format!("http://{s}"), log, _dir: dir };
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("woollamad never published an address; log:\n{}", std::fs::read_to_string(&log).unwrap_or_default());
}

// --- the client -----------------------------------------------------------------------------------

/// The three streaming relays, as (label, model).
fn routes(model: &str) -> [(String, Value); 3] {
    let messages = json!([{"role": "user", "content": "hi"}]);
    [
        (format!("device/{model}"), json!({"model": format!("device/{model}"), "messages": messages, "stream": true})),
        (format!("plain/{model}"), json!({"model": format!("plain/{model}"), "messages": messages, "stream": true})),
        (
            format!("ollama/{model}"),
            json!({"model": format!("ollama/{model}"), "messages": messages, "stream": true,
                   "options": {"num_ctx": 4096}}),
        ),
    ]
}

/// Everything the daemon sent before the body ended, cleanly or not. A body that ends with a
/// transport error is collected up to the error, so the assertions say what was MISSING rather
/// than the test dying on the error.
async fn stream_text(base: &str, body: &Value) -> String {
    let resp = tokio::time::timeout(
        Duration::from_secs(20),
        reqwest::Client::new().post(format!("{base}/v1/chat/completions")).json(body).send(),
    )
    .await
    .expect("woollamad must answer, not hang")
    .expect("request");
    assert_eq!(resp.status().as_u16(), 200, "headers must say 200 for {body}");
    let mut out = Vec::new();
    let mut s = resp.bytes_stream();
    let read = async {
        while let Some(chunk) = s.next().await {
            match chunk {
                Ok(b) => out.extend_from_slice(&b),
                Err(_) => break,
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(20), read).await.expect("the body must end, not hang");
    String::from_utf8_lossy(&out).into_owned()
}

/// The `data:` payloads of an SSE body, in order.
fn data_events(text: &str) -> Vec<String> {
    text.split("\n\n")
        .filter_map(|ev| ev.lines().find_map(|l| l.strip_prefix("data:")).map(|d| d.trim().to_string()))
        .collect()
}

/// The streamed content: every `delta.content`, joined.
fn content(events: &[String]) -> String {
    events
        .iter()
        .filter_map(|e| serde_json::from_str::<Value>(e).ok())
        .filter_map(|v| v["choices"][0]["delta"]["content"].as_str().map(str::to_string))
        .collect()
}

/// A stream that failed must end with an OpenAI-shaped error event and then `[DONE]`, and every
/// event before them must still parse. pi's OpenAI client throws on an `error` event and shows its
/// message; a half frame glued to the error event would instead show "malformed JSON".
fn assert_ends_with_error_event(label: &str, text: &str) {
    let events = data_events(text);
    assert!(events.len() >= 2, "{label}: too few events: {text:?}");
    assert_eq!(events.last().unwrap(), "[DONE]", "{label}: must end with [DONE]: {text:?}");
    let err: Value = serde_json::from_str(&events[events.len() - 2])
        .unwrap_or_else(|_| panic!("{label}: the event before [DONE] must be JSON: {text:?}"));
    assert_eq!(err["error"]["type"], "upstream_error", "{label}: {text:?}");
    assert!(
        err["error"]["message"].as_str().is_some_and(|m| !m.is_empty()),
        "{label}: the error must carry a message: {text:?}"
    );
    for e in &events[..events.len() - 1] {
        assert!(serde_json::from_str::<Value>(e).is_ok(), "{label}: malformed event {e:?} in {text:?}");
    }
    assert!(content(&events).starts_with("t0"), "{label}: the frames before the failure must arrive: {text:?}");
}

// --- the scenarios --------------------------------------------------------------------------------

#[tokio::test]
async fn a_stream_longer_than_the_idle_timeout_is_relayed_whole() {
    let d = spawn_daemon().await;
    for (label, body) in routes("slow") {
        let started = Instant::now();
        let text = stream_text(&d.base, &body).await;
        let events = data_events(&text);
        let want: String = (0..12).map(|i| format!("t{i} ")).collect();
        assert_eq!(content(&events), want, "{label}: every frame must arrive: {text:?}");
        // The point of the scenario: it lasted longer than the timeout and was still not cut.
        assert!(started.elapsed().as_secs_f64() > IDLE_SECS * 2.0, "{label}: the stream ran too briefly to test anything");
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"), "{label}: {text:?}");
        assert!(
            events.iter().any(|e| e.contains("\"finish_reason\":\"stop\"")),
            "{label}: the final chunk must arrive: {text:?}"
        );
        assert!(!text.contains("upstream_error"), "{label}: {text:?}");

        let line = d.stream_end_line(&label).await.unwrap_or_else(|| panic!("{label}: no stream-end line in:\n{}", d.log()));
        assert!(line.contains("outcome=complete"), "{label}: {line}");
        assert!(line.contains(" chunks=") && line.contains(" bytes="), "{label}: {line}");
    }
}

#[tokio::test]
async fn an_upstream_silent_past_the_idle_timeout_ends_with_an_error_event() {
    let d = spawn_daemon().await;
    for (label, body) in routes("silent") {
        let started = Instant::now();
        let text = stream_text(&d.base, &body).await;
        // Bounded by the idle timeout, not by the upstream finally finishing (10 timeouts later).
        assert!(started.elapsed().as_secs_f64() < IDLE_SECS * 6.0, "{label}: took {:?}", started.elapsed());
        assert_ends_with_error_event(&label, &text);

        let line = d.stream_end_line(&label).await.unwrap_or_else(|| panic!("{label}: no stream-end line in:\n{}", d.log()));
        assert!(line.contains("outcome=upstream_error"), "{label}: {line}");
        // The reason must say it was the silence, so nobody goes looking for a crash.
        assert!(line.contains("sent nothing for 1s"), "{label}: the line must carry the reason: {line}");
    }
}

#[tokio::test]
async fn an_upstream_that_drops_mid_frame_ends_with_an_error_event() {
    let d = spawn_daemon().await;
    for (label, body) in routes("drop") {
        let text = stream_text(&d.base, &body).await;
        assert_ends_with_error_event(&label, &text);
        assert!(!text.contains("\"choi\n") && !text.contains("{\"id\":\"x\",\"choi"), "{label}: the half frame must not be relayed: {text:?}");

        let line = d.stream_end_line(&label).await.unwrap_or_else(|| panic!("{label}: no stream-end line in:\n{}", d.log()));
        assert!(line.contains("outcome=upstream_error"), "{label}: {line}");
    }
}

#[tokio::test]
async fn a_client_that_goes_away_is_logged_as_disconnected() {
    let d = spawn_daemon().await;
    for (label, body) in routes("endless") {
        let resp = reqwest::Client::new()
            .post(format!("{}/v1/chat/completions", d.base))
            .json(&body)
            .send()
            .await
            .expect("request");
        let mut s = resp.bytes_stream();
        let first = tokio::time::timeout(Duration::from_secs(10), s.next()).await.expect("first chunk");
        assert!(matches!(first, Some(Ok(_))), "{label}: no first chunk");
        drop(s);

        let line = d.stream_end_line(&label).await.unwrap_or_else(|| panic!("{label}: no stream-end line in:\n{}", d.log()));
        assert!(line.contains("outcome=client_disconnected"), "{label}: {line}");
    }
}
