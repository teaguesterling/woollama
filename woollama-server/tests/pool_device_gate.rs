//! Routes sharing a `management_url` share ONE device-wide gate, across models and endpoints.
//!
//! The startup line says "share one device pool and one concurrency gate", and the homelab config
//! reads that as "the ceiling against the device is 1 in flight". A shared `Arc<Gate>` (asserted by
//! `pool_seeding.rs`) does not make that true if the permit inside it is per model.
//!
//! Why it matters: on the Tiiny the failure is on the single NPU DMA path, not on one model's slot.
//! Beta log #065 (2026-09-03): embedding + chat overlap → `xh2a_pcie_hdma_chan_transfer: timeout`,
//! then a hard reset. 2026-09-14: a chat request overlapping embedding load hung 108 s, returned 200
//! with empty content, and the runtime dropped the chat model. Those overlaps are CROSS-MODEL, which
//! is exactly what a per-model permit lets through.
//!
//! Separate test binary: `WOOLLAMA_CONFIG_DIR` is process-global and would race other files.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use woollama_server::pool::{DeviceModelManager, Gate, PoolError, RestBackend};

const CHAT: &str = "Chat/Model";
const EMBED: &str = "Embed/Model";
const IMAGE: &str = "Image/Model";

/// `build_state` reads the process-global config dir, so tests take turns around that one call.
static ENV: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Default)]
struct DeviceInner {
    running: HashSet<String>,
    /// Inference requests in flight at the device, across EVERY endpoint and model. One counter,
    /// because the thing being protected is one NPU, not one model.
    in_flight: u32,
    max_in_flight: u32,
    /// Every inference request the device received, as (endpoint, model).
    served: Vec<(String, String)>,
}

#[derive(Clone)]
struct DeviceState {
    inner: Arc<StdMutex<DeviceInner>>,
    hold_ms: Arc<AtomicU64>,
}

impl DeviceState {
    /// The sleep is the test's whole sensitivity: without a device-wide gate, requests sit in here
    /// together and `max_in_flight` exceeds 1.
    async fn serve(&self, endpoint: &str, body: &Value) -> Response {
        {
            let mut inner = self.inner.lock().unwrap();
            inner.served.push((endpoint.to_string(), body["model"].as_str().unwrap_or("").to_string()));
            inner.in_flight += 1;
            inner.max_in_flight = inner.max_in_flight.max(inner.in_flight);
        }
        tokio::time::sleep(Duration::from_millis(self.hold_ms.load(Ordering::SeqCst))).await;
        self.inner.lock().unwrap().in_flight -= 1;
        (StatusCode::OK, Json(json!({"ok": true, "choices": [{"message": {"content": "ok"}}], "data": []})))
            .into_response()
    }
}

async fn handle_get(State(st): State<DeviceState>, AxPath(rest): AxPath<String>) -> Response {
    if rest != "running" {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response();
    }
    let running: Vec<String> = st.inner.lock().unwrap().running.iter().cloned().collect();
    (StatusCode::OK, Json(json!({"object": "list", "running": running, "pending": []}))).into_response()
}

async fn handle_post(State(st): State<DeviceState>, AxPath(rest): AxPath<String>) -> Response {
    if let Some(id) = rest.strip_suffix("/start") {
        st.inner.lock().unwrap().running.insert(id.to_string());
        return (StatusCode::OK, Json(json!({"ok": true}))).into_response();
    }
    if let Some(id) = rest.strip_suffix("/stop") {
        st.inner.lock().unwrap().running.remove(id);
        return (StatusCode::OK, Json(json!({"ok": true}))).into_response();
    }
    (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response()
}

async fn handle_chat(State(st): State<DeviceState>, Json(body): Json<Value>) -> Response {
    st.serve("chat", &body).await
}
async fn handle_embeddings(State(st): State<DeviceState>, Json(body): Json<Value>) -> Response {
    st.serve("embeddings", &body).await
}
async fn handle_images(State(st): State<DeviceState>, Json(body): Json<Value>) -> Response {
    st.serve("images", &body).await
}

async fn spawn_router(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

struct Fixture {
    device: DeviceState,
    device_url: String,
    base: String,
    state: Arc<woollama_server::AppState>,
    client: reqwest::Client,
    _cfg: tempfile::TempDir,
}

/// One device with all three models resident, reached by the three routes the homelab config uses
/// for it: chat, embeddings, and images all name the same `management_url`.
async fn boot(hold_ms: u64, queue_timeout: f64) -> Fixture {
    let inner = Arc::new(StdMutex::new(DeviceInner {
        running: [CHAT, EMBED, IMAGE].iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }));
    let device = DeviceState { inner, hold_ms: Arc::new(AtomicU64::new(hold_ms)) };
    let device_url = spawn_router(
        Router::new()
            .route("/api/v1/models/{*rest}", get(handle_get).post(handle_post))
            .route("/chat/completions", post(handle_chat))
            .route("/embeddings", post(handle_embeddings))
            .route("/images/generations", post(handle_images))
            .with_state(device.clone()),
    )
    .await;

    let cfg = tempfile::tempdir().unwrap();
    std::fs::write(cfg.path().join("recipes.toml"), "").unwrap();
    std::fs::write(cfg.path().join("mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
    let route = |name: &str| {
        format!(
            "[inferencers.{name}]\nbase_url=\"{device_url}\"\nmanagement_url=\"{device_url}\"\n\
             parallel=1\nqueue_timeout={queue_timeout}\n"
        )
    };
    std::fs::write(
        cfg.path().join("inferencers.toml"),
        [route("tiiny"), route("tiiny-embed"), route("tiiny-other")].concat(),
    )
    .unwrap();

    let state = {
        let _env = ENV.lock().await;
        std::env::set_var("WOOLLAMA_CONFIG_DIR", cfg.path());
        std::env::set_var("WOOLLAMA_STATE_DIR", cfg.path());
        Arc::new(woollama_server::build_state().await)
    };
    let base = spawn_router(woollama_server::router(state.clone())).await;
    Fixture { device, device_url, base, state, client: reqwest::Client::new(), _cfg: cfg }
}

impl Fixture {
    fn send(&self, path: &'static str, body: Value) -> tokio::task::JoinHandle<reqwest::Response> {
        let (c, base) = (self.client.clone(), self.base.clone());
        tokio::spawn(async move { c.post(format!("{base}{path}")).json(&body).send().await.unwrap() })
    }
    fn chat(&self) -> tokio::task::JoinHandle<reqwest::Response> {
        self.send("/v1/chat/completions", json!({"model": format!("tiiny/{CHAT}"), "messages": []}))
    }
    fn embed(&self) -> tokio::task::JoinHandle<reqwest::Response> {
        self.send("/v1/embeddings", json!({"model": format!("tiiny-embed/{EMBED}"), "input": ["a"]}))
    }
    fn image(&self) -> tokio::task::JoinHandle<reqwest::Response> {
        self.send("/v1/images/generations", json!({"model": format!("tiiny-other/{IMAGE}"), "prompt": "x"}))
    }
    fn served(&self) -> Vec<(String, String)> {
        self.device.inner.lock().unwrap().served.clone()
    }
    fn max_in_flight(&self) -> u32 {
        self.device.inner.lock().unwrap().max_in_flight
    }
    fn queued(&self, model: &str) -> u32 {
        self.state.pools.get("tiiny").unwrap().0.queued(model)
    }
    async fn wait_until(&self, what: &str, cond: impl Fn(&Self) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !cond(self) {
            assert!(Instant::now() < deadline, "timed out waiting until {what}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

/// The contract itself. Chat, embeddings and images on three routes onto one device: the device
/// sees one at a time. A per-model permit gives 2 here; an ungated image route makes it 3.
#[tokio::test]
async fn chat_embeddings_and_images_never_overlap_at_the_device() {
    let f = boot(200, 30.0).await;

    let (chat, embed, image) = (f.chat(), f.embed(), f.image());
    assert_eq!(chat.await.unwrap().status(), 200);
    assert_eq!(embed.await.unwrap().status(), 200);
    assert_eq!(image.await.unwrap().status(), 200);

    assert_eq!(
        f.max_in_flight(),
        1,
        "routes sharing a management_url must share one device-wide gate; the device saw {} \
         concurrent inference requests across models, which is the overlap in beta log #065",
        f.max_in_flight()
    );
    assert_eq!(f.served().len(), 3, "every request should still be served: {:?}", f.served());
}

/// A long chat must not make an embedding hang: it waits `queue_timeout`, then answers 503 with
/// `Retry-After`, while the chat is still running. A client can act on that; it cannot act on a hang.
#[tokio::test]
async fn a_request_behind_another_model_is_refused_at_queue_timeout() {
    let f = boot(1500, 0.3).await;

    let chat = f.chat();
    f.wait_until("the chat reaches the device", |f| !f.served().is_empty()).await;

    let started = Instant::now();
    let refused = f.embed().await.unwrap();
    let waited = started.elapsed();

    assert_eq!(refused.status(), 503, "must be refused, not overlapped with the chat");
    assert!(refused.headers().contains_key("retry-after"), "503 must carry Retry-After");
    assert!(waited >= Duration::from_millis(250), "must wait before refusing; waited {waited:?}");
    assert!(waited < Duration::from_millis(1200), "must not outlast the chat; waited {waited:?}");
    assert_eq!(f.served().len(), 1, "the refused embedding must never reach the device");

    assert_eq!(chat.await.unwrap().status(), 200, "the chat itself is unaffected");
}

/// A client that gives up while waiting behind another model leaves nothing behind: no queued
/// count (which would pin the model against eviction forever), no permit, and nothing sent to the
/// device for a caller who is gone.
#[tokio::test]
async fn a_disconnected_waiter_holds_nothing_and_never_reaches_the_device() {
    let f = boot(600, 30.0).await;

    let chat = f.chat();
    f.wait_until("the chat reaches the device", |f| !f.served().is_empty()).await;

    let abandoned = f.embed();
    f.wait_until("the embedding queues behind the chat", |f| f.queued(EMBED) == 1).await;
    abandoned.abort();
    f.wait_until("the abandoned request releases its place", |f| f.queued(EMBED) == 0).await;

    assert_eq!(chat.await.unwrap().status(), 200);

    // The gate must still work: a later embedding gets through, so no permit leaked.
    f.device.hold_ms.store(0, Ordering::SeqCst);
    assert_eq!(f.embed().await.unwrap().status(), 200, "a permit leaked to the abandoned waiter");
    assert_eq!(
        f.served().iter().filter(|(e, _)| e == "embeddings").count(),
        1,
        "only the live embedding should reach the device: {:?}",
        f.served()
    );
}

fn gate(device_url: &str, parallel: u32, queue_timeout: f64) -> (Arc<DeviceModelManager>, Arc<Gate>) {
    let m = Arc::new(DeviceModelManager::with_retry_after(
        Arc::new(RestBackend::device(device_url.to_string(), Default::default(), 0.01, 5.0)),
        5.0,
    ));
    let g = Arc::new(Gate::new(m.clone(), parallel, None, queue_timeout, None, 5.0));
    (m, g)
}

/// Per-model queues feed the device gate, so a bulk client cannot put a chat request behind its
/// whole backlog: the chat waits for the one request in flight, not for every queued embedding.
/// This is the reason the per-model permit stays alongside the device permit.
#[tokio::test]
async fn a_backlog_on_one_model_does_not_stand_in_front_of_another() {
    let f = boot(0, 30.0).await;
    let (m, g) = gate(&f.device_url, 1, 10.0);
    let order: Arc<StdMutex<Vec<&'static str>>> = Arc::default();

    let held = g.enter(EMBED).await.unwrap();

    let spawn_waiter = |model: &'static str, tag: &'static str| {
        let (g, order) = (g.clone(), order.clone());
        tokio::spawn(async move {
            let slot = g.enter(model).await.unwrap();
            order.lock().unwrap().push(tag);
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(slot);
        })
    };

    let backlog = [spawn_waiter(EMBED, "embed-2"), spawn_waiter(EMBED, "embed-3")];
    while m.queued(EMBED) < 2 {
        tokio::task::yield_now().await;
    }
    // Let the backlog settle into its queue before the chat arrives last.
    tokio::time::sleep(Duration::from_millis(30)).await;
    let chat = spawn_waiter(CHAT, "chat");
    // A sleep, not a spin on `queued(CHAT)`: a chat that is NOT gated never queues, and that must
    // fail the assertion below rather than hang the test.
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(order.lock().unwrap().is_empty(), "nothing may run while the first embedding is held");

    drop(held);
    chat.await.unwrap();
    for t in backlog {
        t.await.unwrap();
    }

    assert_eq!(
        order.lock().unwrap().first().copied(),
        Some("chat"),
        "the chat arrived last but should run next: {:?}",
        order.lock().unwrap()
    );
}

/// `queue_timeout` bounds the whole wait, not each permit in turn. Otherwise adding a second
/// permit silently doubles a documented bound.
#[tokio::test]
async fn queue_timeout_bounds_both_permits_together() {
    let f = boot(0, 30.0).await;
    let (_m, g) = gate(&f.device_url, 1, 0.4);

    // Holds the embed model permit and the device permit.
    let first = g.enter(EMBED).await.unwrap();

    // Waits on the device permit, and will get it once `first` lets go.
    let g2 = g.clone();
    let device_taker = tokio::spawn(async move {
        let slot = g2.enter(CHAT).await.unwrap();
        tokio::time::sleep(Duration::from_millis(600)).await;
        drop(slot);
    });
    tokio::time::sleep(Duration::from_millis(20)).await;

    // Waits on the embed model permit, then on the device permit `device_taker` will hold by then.
    let g3 = g.clone();
    let started = Instant::now();
    let measured = tokio::spawn(async move {
        let outcome = g3.enter(EMBED).await.map(drop);
        (outcome, started.elapsed())
    });

    tokio::time::sleep(Duration::from_millis(300)).await;
    drop(first);

    let (outcome, waited) = measured.await.unwrap();
    assert!(matches!(outcome, Err(PoolError::Backpressure(_))), "expected Backpressure, got {outcome:?}");
    assert!(
        waited < Duration::from_millis(600),
        "waited {waited:?}: queue_timeout (0.4s) must cover both permits, not 0.4s each"
    );
    device_taker.await.unwrap();
}
