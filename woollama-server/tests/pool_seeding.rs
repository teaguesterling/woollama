//! Issue #26: `<provider>/default` must resolve against the DEVICE's residency, not against
//! woollamad's own bookkeeping.
//!
//! The ownership model matters more than the symptom. woollama is NOT the only consumer of a
//! device's management API — the vendor's own UI loads models, and any caller needing endpoints
//! woollama doesn't serve (images, embeddings, ASR) must drive the device directly. So our
//! `loaded` flags are a cache of someone else's state; the device is the only thing that knows.
//! Seeding once at startup would have fixed only the cold-start instance of this.
//!
//! `docs/configuration.md` says `default` resolves "to whichever model is currently loaded".
//! Before this fix it resolved against `DeviceModelManager`'s local `entries` map, which starts
//! empty on every restart and only records models *this instance* loaded — so a model resident on
//! the device but loaded by someone else was invisible, and `<provider>/default` either 400'd or
//! fell through to the `[virtual]` table and evicted a perfectly good model to load its entry.
//!
//! Reported from real hardware by the `Tiiny` session; the discriminating setup is theirs:
//! start against a device with a model already resident that woollamad did not load.

mod common;

use common::{spawn_rest, IdLoc, RestMockConfig, RunningShape};
use woollama_server::pool::{DeviceModelManager, RestBackend};

/// Serializes every test in this binary that drives `build_state()`. `WOOLLAMA_CONFIG_DIR` is
/// process-global, so two concurrent builds read each other's config dir. Same guard as
/// tests/http_downstream.rs — it belongs anywhere `build_state` is called from more than one test.
static ENV: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn device_cfg() -> RestMockConfig {
    RestMockConfig {
        running_route: "/api/v1/models/running".to_string(),
        start_route: "/api/v1/models/{id}/start".to_string(),
        stop_route: "/api/v1/models/{id}/stop".to_string(),
        running_shape: RunningShape::Strings { field: "running".to_string() },
        id_loc: IdLoc::Path,
    }
}

fn manager(base_url: &str) -> DeviceModelManager {
    DeviceModelManager::new(std::sync::Arc::new(RestBackend::device(
        base_url.to_string(),
        Default::default(),
        0.05,
        5.0,
    )))
}

#[tokio::test]
async fn a_model_resident_but_not_loaded_by_us_is_visible() {
    let device = spawn_rest(device_cfg());
    // The device is already running a model. woollamad did not load it — this is the state after
    // a woollamad restart, or when anything else on the box drives the device.
    device.set_loaded(&["Qwen/Qwen3.6-35B-A3B-turbo"]);

    let mgr = manager(&device.base_url);

    // The bug: local bookkeeping is empty, so `default` has nothing to resolve to.
    assert!(
        mgr.snapshot().is_empty(),
        "precondition: local bookkeeping starts empty — that is exactly the problem"
    );

    assert_eq!(
        mgr.residency().await.models,
        vec!["Qwen/Qwen3.6-35B-A3B-turbo".to_string()],
        "the device's actual residency is what `default` resolves against"
    );
}

#[tokio::test]
async fn residency_tracks_a_change_made_by_someone_else_after_we_started() {
    // THE property, and the one seeding could never have provided: another consumer changes what
    // is resident while we are running. Nothing tells us; only asking the device again can.
    let device = spawn_rest(device_cfg());
    device.set_loaded(&["First"]);
    let mgr = manager(&device.base_url);
    let mgr = mgr.with_residency_ttl(0.0); // no coalescing: read every time
    assert_eq!(mgr.residency().await.models, vec!["First".to_string()]);

    // The desktop app / a direct `tiiny image` call swaps the device out from under us.
    device.set_loaded(&["Second"]);
    let seen = mgr.residency().await.models;
    assert_eq!(
        seen,
        vec!["Second".to_string()],
        "a residency change we did not make must be visible — the device is the only thing that knows"
    );
}

#[tokio::test]
async fn concurrent_reads_coalesce_into_one_device_query() {
    let device = spawn_rest(device_cfg());
    device.set_loaded(&["A"]);
    let mgr = manager(&device.base_url);

    for _ in 0..5 {
        mgr.residency().await;
    }

    // One query, not five. The freshness window is a COALESCING window — a burst of `default`
    // requests shares one device round trip — not a claim that our view stays correct between
    // reads. (`residency_tracks_a_change_made_by_someone_else` pins the other half.)
    assert_eq!(
        device.requests_to("/running").len(),
        1,
        "a burst of reads inside the window must share one query"
    );
}

#[tokio::test]
async fn a_failed_read_is_reported_as_not_current() {
    // An empty set has two causes that justify OPPOSITE advice: the query succeeded and nothing
    // matched (the operator's expectations are wrong), or the query failed (we are blind).
    // Conflating them produced a confident "fix your config" while the config was fine and three
    // models were resident — the caller just had a 401. Reported from a real migration.
    let unreachable = manager("http://127.0.0.1:1");
    let r = unreachable.residency().await;
    assert!(!r.current, "a failed device query must not present its empty result as fact");
    assert!(r.models.is_empty());
    assert!(r.capabilities.is_empty(), "a failed read reports no capabilities either");

    let device = spawn_rest(device_cfg());
    device.set_loaded(&["A"]);
    let ok = manager(&device.base_url);
    let r = ok.residency().await;
    assert!(r.current, "a successful query is current");
    assert_eq!(r.models, vec!["A".to_string()]);
}

#[tokio::test]
async fn an_unreachable_device_does_not_make_a_read_fatal() {
    // Seeding is best-effort: if the device is down, the request that follows will fail on its
    // own and produce a real error. Seeding must not turn that into a different, earlier error —
    // nor panic, since it runs on the request path.
    let mgr = manager("http://127.0.0.1:1");
    mgr.residency().await;
    assert!(mgr.snapshot().is_empty(), "an unreachable device leaves the pool empty, not poisoned");

    // And it must not latch: once the device is reachable, a later attempt should still seed.
    let device = spawn_rest(device_cfg());
    device.set_loaded(&["B"]);
    let mgr2 = manager(&device.base_url);
    mgr2.residency().await;
    assert_eq!(mgr2.snapshot(), vec!["B".to_string()]);
}

// --- end-to-end: the exact symptom reported from hardware -----------------------------------

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

#[derive(Clone)]
struct Dev(Arc<Mutex<Vec<String>>>);

/// A device that is already running `resident`, plus enough of an OpenAI surface to dispatch.
async fn spawn_device(resident: &str) -> (String, Dev) {
    let state = Dev(Arc::new(Mutex::new(vec![resident.to_string()])));
    let s2 = state.clone();
    let router = Router::new()
        .route(
            "/api/v1/models/running",
            get(|State(d): State<Dev>| async move { Json(json!({"running": *d.0.lock().unwrap()})) }),
        )
        .route(
            "/v1/chat/completions",
            post(|Json(b): Json<Value>| async move {
                // Echo the model actually dispatched, so the test can prove WHICH one was used.
                Json(json!({"choices": [{"message": {"role": "assistant",
                    "content": b["model"].as_str().unwrap_or("?")}}]}))
            }),
        )
        .with_state(s2);
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
    (format!("http://{addr}"), state)
}

/// Tiiny's repro, end to end: a model resident on the device that woollamad did not load, no
/// `[virtual]` table configured. Before #26 this returned
/// `400 "model 'default' requested but no model is loaded ..."` — with the model loaded.
#[tokio::test]
async fn default_serves_a_resident_model_with_no_virtual_table() {
    let (url, _dev) = spawn_device("Qwen/Qwen3.6-35B-A3B-turbo").await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    // NOTE: no [inferencers.dev.virtual] table — that absence is the point.
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!("[inferencers.dev]\nbase_url=\"{url}/v1\"\nmanagement_url=\"{url}\"\n"),
    )
    .unwrap();
    let _env = ENV.lock().await;
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_STATE_DIR", cfg.path());
    let state = Arc::new(woollama_server::build_state().await);
    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("WOOLLAMA_STATE_DIR");

    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, woollama_server::router(state)).await.unwrap() });

    let r = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({"model": "dev/default", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();

    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "`default` must resolve to the resident model, got {body}");
    // Assert WHICH model was dispatched — a 200 alone would also pass if it silently picked
    // something else.
    assert_eq!(
        body["choices"][0]["message"]["content"], "Qwen/Qwen3.6-35B-A3B-turbo",
        "the resident model is what `default` must mean"
    );

    // --- Case 2 (SAME test fn): shared pool + gate for one device -------------------------
    // Both cases drive build_state via the process-global WOOLLAMA_CONFIG_DIR, so they cannot be
    // separate parallel #[tokio::test]s — the second would swap the config dir out from under the
    // first. (That is exactly how this first failed.)
    //
    // Two inferencers naming the same `management_url` must share one pool AND one gate.
    // 
    // Reported from hardware: alternating between two routes onto one device swapped the model on
    // every call (~30s each way), because each route's pool was always empty from the other's
    // perspective. The gate half matters more: `parallel` is per-inferencer, so two routes onto one
    // device previously enforced it twice over and **doubled** the effective in-flight concurrency —
    // on hardware where `parallel = 1` exists to prevent a wedge, that is a safety property being
    // silently halved.

    let (url, _dev) = spawn_device("Resident/Model").await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!(
            "[inferencers.fast]\nbase_url=\"{url}/v1\"\nmanagement_url=\"{url}\"\nparallel=4\n\
             [inferencers.slow]\nbase_url=\"{url}/v1\"\nmanagement_url=\"{url}\"\nparallel=1\n"
        ),
    )
    .unwrap();
    // No ENV lock here: this fn already holds it from Case 1. `let _env = ...` a second time
    // SHADOWS the binding without dropping the first, so re-acquiring deadlocks on a
    // non-reentrant mutex — which is exactly what happened when the guard was added mechanically.
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_STATE_DIR", cfg.path());
    let state = Arc::new(woollama_server::build_state().await);
    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("WOOLLAMA_STATE_DIR");

    let (fast_mgr, fast_gate) = state.pools.get("fast").expect("fast has a pool");
    let (slow_mgr, slow_gate) = state.pools.get("slow").expect("slow has a pool");

    assert!(
        Arc::ptr_eq(fast_mgr, slow_mgr),
        "both routes must observe ONE residency view, or each swaps the device out from under the other"
    );
    assert!(
        Arc::ptr_eq(fast_gate, slow_gate),
        "both routes must share ONE gate, or `parallel` is enforced once per route and the device \
         sees double the configured concurrency"
    );

    // The shared gate takes the most restrictive limit, not whichever inferencer sorted first.
    // Asserted via behaviour rather than a private field: seeding through one route is visible
    // from the other, which is only true if the manager really is shared.
    fast_mgr.residency().await;
    assert_eq!(
        slow_mgr.snapshot(),
        vec!["Resident/Model".to_string()],
        "a load seen by one route must be visible to the other"
    );
}


/// End to end against the device's REAL running-payload shape (issue #20): a bare-string
/// `running` array plus a sibling `instances.running[]` of objects carrying `capabilities`.
/// Verbatim from a live device, not invented — inferring a vendor's payload is the trap this
/// whole line of work exists to avoid.
///
/// Asserts the WIRING, not the predicate: a capability rule that computed correctly but was never
/// consulted by `default` resolution would pass a unit test and fail here.
#[tokio::test]
async fn default_skips_an_embedder_using_the_devices_own_capability_report() {
    let state = Arc::new(Mutex::new(()));
    let _ = state;
    let router = Router::new()
        .route(
            "/api/v1/models/running",
            get(|| async {
                Json(json!({
                    "object": "list",
                    "running": ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3.6-35B-A3B-turbo"],
                    "pending": [],
                    "instances": {
                        "running": [
                            {"model_id": "Qwen/Qwen3-Embedding-0.6B", "status": "running",
                             "type": "Text Embedding", "capabilities": ["embedding"], "output": "Vector"},
                            {"model_id": "Qwen/Qwen3.6-35B-A3B-turbo", "status": "running",
                             "type": "Image-Text-to-Text", "capabilities": ["main"], "output": "Text"}
                        ],
                        "pending": []
                    }
                }))
            }),
        )
        .route(
            "/v1/chat/completions",
            post(|Json(b): Json<Value>| async move {
                Json(json!({"choices": [{"message": {"role": "assistant",
                    "content": b["model"].as_str().unwrap_or("?")}}]}))
            }),
        );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
    let url = format!("http://{addr}");

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    // No `models` list and no `[capabilities]` table — the whole point is that this needs NO
    // configuration.
    //
    // FIXTURE CONSTRAINT: the embedder must sort BEFORE the chat model, or this passes without
    // the capability filter doing anything. Fallback ordering is lexicographic, so a chat model
    // whose id happens to sort first wins by luck of the alphabet — a real hardware test nearly
    // shipped as "verified" for exactly that reason.
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!("[inferencers.dev]\nbase_url=\"{url}/v1\"\nmanagement_url=\"{url}\"\n"),
    )
    .unwrap();
    let _env = ENV.lock().await;
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_STATE_DIR", cfg.path());
    let st = Arc::new(woollama_server::build_state().await);
    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("WOOLLAMA_STATE_DIR");

    let rl = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let raddr = rl.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(rl, woollama_server::router(st)).await.unwrap() });

    let r = reqwest::Client::new()
        .post(format!("http://{raddr}/v1/chat/completions"))
        .json(&json!({"model": "dev/default", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["choices"][0]["message"]["content"], "Qwen/Qwen3.6-35B-A3B-turbo",
        "`default` must skip the embedder the device itself labelled `embedding`, with no config"
    );
}


/// Pre-dispatch validation and `/v1/models` annotation, end to end (issue #20).
///
/// Asserts the WIRING: a rule that computes correctly but is never consulted by the endpoint
/// would pass its unit test and fail here. The backend deliberately answers 200 to anything, so a
/// request that reaches it proves the check did NOT run.
#[tokio::test]
async fn a_declared_embedder_is_refused_on_chat_and_annotated_in_v1_models() {
    let router = Router::new()
        .route("/api/v1/models/running", get(|| async { Json(json!({"running": []})) }))
        .route("/v1/chat/completions", post(|| async { Json(json!({"choices": [{"message": {"role": "assistant", "content": "REACHED BACKEND"}}]})) }));
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
    let url = format!("http://{addr}");

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!(
            // No management_url: this exercises the PLAIN pass-through path, proving the check
            // is not pool-only. It was, initially — the test caught it.
            "[inferencers.dev]\nbase_url=\"{url}/v1\"\n\
             models=[\"Qwen/Qwen3-Embedding-0.6B\",\"Qwen/Chat-7B\"]\n\
             [inferencers.dev.capabilities]\nembedding=[\"*Embedding*\"]\n"
        ),
    )
    .unwrap();
    let _env = ENV.lock().await;
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_STATE_DIR", cfg.path());
    let st = Arc::new(woollama_server::build_state().await);
    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("WOOLLAMA_STATE_DIR");

    let rl = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let raddr = rl.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(rl, woollama_server::router(st)).await.unwrap() });
    let c = reqwest::Client::new();

    // The embedder must not reach the backend on the chat path.
    let r = c
        .post(format!("http://{raddr}/v1/chat/completions"))
        .json(&json!({"model": "dev/Qwen/Qwen3-Embedding-0.6B", "messages": []}))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 400, "expected a refusal, got {body}");
    let msg = body["error"]["message"].as_str().unwrap_or("");
    assert!(msg.contains("embedding"), "the error must say what the model IS: {msg}");
    assert!(!msg.contains("REACHED BACKEND"));

    // An undeclared model on the same inferencer is unaffected.
    let r = c
        .post(format!("http://{raddr}/v1/chat/completions"))
        .json(&json!({"model": "dev/Qwen/Chat-7B", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "an undeclared model must dispatch exactly as before");

    // /v1/models annotates what is declared, and says nothing about what isn't.
    let models: Value = c.get(format!("http://{raddr}/v1/models")).send().await.unwrap().json().await.unwrap();
    let entries = models["data"].as_array().unwrap();
    let emb = entries.iter().find(|m| m["id"] == "dev/Qwen/Qwen3-Embedding-0.6B").expect("embedder listed");
    assert_eq!(emb["capabilities"], json!(["embedding"]));
    let chat = entries.iter().find(|m| m["id"] == "dev/Qwen/Chat-7B").expect("chat model listed");
    assert!(chat.get("capabilities").is_none(), "undeclared means absent, never 'cannot': {chat}");
}

/// `/v1/models` must answer "which model will respond right now", not just "which exist".
///
/// Reported by a caller who was blocked on exactly this: `/v1/models` listed a model, the request
/// 503'd with "not loaded", and there was no way to tell the two states apart except by sending a
/// request that may fail after a thirty-second load.
#[tokio::test]
async fn v1_models_reports_what_is_actually_loaded() {
    let router = Router::new()
        .route(
            "/api/v1/models/running",
            // Resident: one DECLARED model, and one the operator never listed.
            get(|| async {
                Json(json!({"running": ["Declared/Up", "Surprise/AlsoUp"],
                            "instances": {"running": []}}))
            }),
        )
        .route("/v1/chat/completions", post(|| async { Json(json!({"choices": []})) }));
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
    let url = format!("http://{addr}");

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!(
            "[inferencers.dev]\nbase_url=\"{url}/v1\"\nmanagement_url=\"{url}\"\n\
             models=[\"Declared/Up\",\"Declared/Down\"]\n\
             [inferencers.plain]\nbase_url=\"{url}/v1\"\nmodels=[\"Plain/Model\"]\n"
        ),
    )
    .unwrap();
    let _env = ENV.lock().await;
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_STATE_DIR", cfg.path());
    let st = Arc::new(woollama_server::build_state().await);
    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("WOOLLAMA_STATE_DIR");

    let rl = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let raddr = rl.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(rl, woollama_server::router(st)).await.unwrap() });

    let body: Value = reqwest::get(format!("http://{raddr}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let find = |id: &str| {
        body["data"].as_array().unwrap().iter().find(|m| m["id"] == id).cloned()
    };

    // Declared AND resident -> loaded true. Declared but not resident -> loaded false. That
    // distinction is the whole point; before it, both looked identical.
    assert_eq!(find("dev/Declared/Up").expect("listed")["loaded"], json!(true));
    assert_eq!(find("dev/Declared/Down").expect("listed")["loaded"], json!(false));

    // Resident but never declared: still routable, still the thing that will answer, so it is
    // listed — flagged, because the operator did not promise it.
    let surprise = find("dev/Surprise/AlsoUp").expect("a resident model must be discoverable");
    assert_eq!(surprise["loaded"], json!(true));
    assert_eq!(surprise["undeclared"], json!(true));

    // An inferencer with NO pool cannot know, so it says nothing. Absent must not read as "no".
    let plain = find("plain/Plain/Model").expect("listed");
    assert!(plain.get("loaded").is_none(), "unknown must not be reported as not-loaded: {plain}");
}

/// Issue #38: the pool must recover when a model is unloaded underneath it.
///
/// Reported from hardware: certain requests crashed a model instance, and every subsequent request
/// then returned "…is not loaded" INDEFINITELY. `ensure_loaded` short-circuits on its own belief
/// before consulting the backend, so once that belief went stale it stayed stale for the life of
/// the process. A 20-item batch lost items 15–20 to it.
///
/// The fix asks the backend rather than parsing its error: whether a given message means
/// "unloaded" is vendor wording, but whether the model is running is a question the backend can
/// answer directly. This test drives that end to end — the device drops the model AND fails the
/// call, and the NEXT request must succeed.
#[tokio::test]
async fn a_model_dropped_underneath_the_pool_is_reloaded_on_the_next_request() {
    #[derive(Clone)]
    struct Dev {
        resident: Arc<Mutex<Vec<String>>>,
        /// Fail the next chat call and drop the model, simulating a crash that unloads it.
        crash_next: Arc<Mutex<bool>>,
        starts: Arc<Mutex<usize>>,
    }
    let dev = Dev {
        resident: Arc::new(Mutex::new(vec!["M".to_string()])),
        crash_next: Arc::new(Mutex::new(true)),
        starts: Arc::new(Mutex::new(0)),
    };

    let d = dev.clone();
    let router = Router::new()
        .route(
            "/api/v1/models/running",
            get({
                let d = d.clone();
                move || {
                    let d = d.clone();
                    async move { Json(json!({"running": *d.resident.lock().unwrap()})) }
                }
            }),
        )
        .route(
            "/api/v1/models/{id}/start",
            post({
                let d = d.clone();
                move |axum::extract::Path(id): axum::extract::Path<String>| {
                    let d = d.clone();
                    async move {
                        *d.starts.lock().unwrap() += 1;
                        d.resident.lock().unwrap().push(id);
                        Json(json!({"ok": true}))
                    }
                }
            }),
        )
        .route(
            "/v1/chat/completions",
            post({
                let d = d.clone();
                move || {
                    let d = d.clone();
                    async move {
                        let mut crash = d.crash_next.lock().unwrap();
                        if *crash {
                            *crash = false;
                            // The crash. CRITICALLY, the model stays in the running list — real
                            // hardware keeps reporting a crashed instance as `running` (with a
                            // stale in-flight count) for 2-5s until reaping catches up. An earlier
                            // fixture cleared it immediately, which made a reconcile-based fix
                            // look correct when on real hardware it would have done nothing.
                            return (
                                axum::http::StatusCode::BAD_GATEWAY,
                                Json(json!({"error": {"message": "Upstream model server request failed"}})),
                            );
                        }
                        (axum::http::StatusCode::OK,
                         Json(json!({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})))
                    }
                }
            }),
        );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
    let url = format!("http://{addr}");

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        format!("[inferencers.dev]\nbase_url=\"{url}/v1\"\nmanagement_url=\"{url}\"\n"),
    )
    .unwrap();
    let _env = ENV.lock().await;
    std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
    std::env::set_var("WOOLLAMA_STATE_DIR", cfg.path());
    let st = Arc::new(woollama_server::build_state().await);
    std::env::remove_var("WOOLLAMA_CONFIG_DIR");
    std::env::remove_var("WOOLLAMA_STATE_DIR");

    let rl = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let raddr = rl.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(rl, woollama_server::router(st)).await.unwrap() });
    let c = reqwest::Client::new();
    let call = || {
        c.post(format!("http://{raddr}/v1/chat/completions"))
            .json(&json!({"model": "dev/M", "messages": [{"role": "user", "content": "hi"}]}))
            .send()
    };

    // First call crashes the instance and unloads the model. The failure itself is expected.
    let r = call().await.unwrap();
    assert!(r.status().is_server_error(), "the crash is relayed, got {}", r.status());

    // THE POINT: the next call must succeed. Before the fix this returned the same failure
    // forever, because the pool still believed the model was resident and never asked again.
    let r = call().await.unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "the pool must recover, not fail identically forever: {body}");
    assert_eq!(body["choices"][0]["message"]["content"], "ok");
    assert!(
        *dev.starts.lock().unwrap() >= 1,
        "the model must actually have been RELOADED — recovery cannot come from trusting the \
         backend's list, which still contains the dead instance at this point"
    );
}
