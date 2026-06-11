//! woollama-core (Rust) — the PyO3 wheel: `import woollama.core`.
//!
//! This crate is now a THIN WRAPPER over `woollama-engine` (the pure-Rust engine).
//! It owns only the Python boundary:
//!   - `InferenceError` (the `EngineError` pyclass) + `EngineError -> PyErr`,
//!   - `PyToolProvider` — bridges a Python `ToolProvider` callback to the engine's
//!     Rust `ToolProvider` trait (the `pyo3_async_runtimes` coroutine bridge),
//!   - the async-iterator pyclasses (`EventIter`, `DeltaStream`) wrapping engine streams,
//!   - `ModelRegistry` wrapping `engine::Registry`,
//!   - the pyfunctions (depythonize args → engine → pythonize results).
//!
//! All inference/orchestration logic lives in `woollama-engine`; its conformance
//! suite (run against this wheel) is the oracle. Behavior mirrors `woollama.core` in
//! Python.

use std::pin::Pin;
use std::sync::Arc;

use futures::stream::{Stream, StreamExt};
use pyo3::exceptions::{PyException, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use woollama_engine as engine;
use engine::{EngineError, Event, ToolProvider};

// --- InferenceError pyclass + EngineError -> PyErr ----------------------------

/// Structured inference/orchestration error — mirrors the Python
/// `core.InferenceError(message, kind, status, payload=None)`. The router re-exports
/// it as `OrchestrationError` and maps `.kind`/`.status` to its surface; `.payload`
/// carries the raw upstream response.
#[pyclass(extends = PyException, subclass)]
struct InferenceError {
    #[pyo3(get)]
    message: String,
    #[pyo3(get)]
    kind: Option<String>,
    #[pyo3(get)]
    status: Option<i64>,
    #[pyo3(get)]
    payload: Option<Py<PyAny>>,
}

#[pymethods]
impl InferenceError {
    #[new]
    #[pyo3(signature = (message, kind=None, status=None, payload=None))]
    fn new(message: String, kind: Option<String>, status: Option<i64>, payload: Option<Py<PyAny>>) -> Self {
        InferenceError { message, kind, status, payload }
    }

    // NB: like `BaseException`, construct POSITIONALLY — a `payload=`-style keyword
    // reaches the inherited `BaseException.__init__` (PyO3 wires `#[new]` as __new__
    // only), which rejects kwargs. `engine_err` always builds it positionally.
    fn __str__(&self) -> String {
        self.message.clone()
    }
}

/// Convert a pure-engine `EngineError` into the raised `InferenceError` pyclass.
fn engine_err(e: EngineError) -> PyErr {
    Python::with_gil(|py| {
        let payload_py = e
            .payload
            .and_then(|p| pythonize::pythonize(py, &p).ok())
            .map(|b| b.unbind());
        match py
            .get_type::<InferenceError>()
            .call1((e.message, e.kind, e.status, payload_py))
        {
            Ok(exc) => PyErr::from_value(exc),
            Err(err) => err,
        }
    })
}

// --- Python <-> serde helpers -------------------------------------------------

fn pyval(v: &Value) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| Ok(pythonize::pythonize(py, v)?.unbind()))
}

fn depy(v: &Option<Bound<'_, PyAny>>) -> PyResult<Option<Value>> {
    v.as_ref().map(pythonize::depythonize).transpose().map_err(Into::into)
}

/// `"{TypeName}: {message}"` for a caught Python error — matches the Python loop's
/// `f"{type(e).__name__}: {e}"` when a dispatch raises (orchestrate.py:141).
fn pyerr_brief(e: &PyErr) -> String {
    Python::with_gil(|py| {
        let tn = e
            .get_type(py)
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "Error".to_string());
        let val = e.value(py).str().map(|s| s.to_string()).unwrap_or_default();
        format!("{tn}: {val}")
    })
}

/// Render a Python `ToolResult` (duck-typed: `.blocks` list, `.is_error` bool) into
/// `(content, is_error)` — the `tool` message content per `tooling.render_tool_result`
/// for text-only `DEFAULT_CAPS`, plus the `is_error` flag.
fn render_tool_result_rs(result: &Bound<'_, PyAny>) -> PyResult<(String, bool)> {
    let is_error: bool = result.getattr("is_error").and_then(|v| v.extract()).unwrap_or(false);
    let blocks = result.getattr("blocks")?;
    let mut text_parts: Vec<String> = Vec::new();
    let mut dumped: Vec<Value> = Vec::new();
    let mut any_block = false;
    for b in blocks.try_iter()? {
        let b = b?;
        any_block = true;
        if let Ok(t) = b.getattr("text") {
            if let Ok(s) = t.extract::<String>() {
                text_parts.push(s);
                continue;
            }
        }
        let v: Value = match b.getattr("model_dump") {
            Ok(md) if md.is_callable() => md
                .call0()
                .ok()
                .and_then(|d| pythonize::depythonize(&d).ok())
                .unwrap_or(Value::Null),
            _ => pythonize::depythonize(&b).unwrap_or(Value::Null),
        };
        dumped.push(v);
    }
    let mut body = if !text_parts.is_empty() {
        text_parts.join("\n")
    } else if any_block {
        serde_json::to_string(&dumped).unwrap_or_default()
    } else {
        String::new()
    };
    if is_error {
        body = if body.is_empty() { "[tool error]".to_string() } else { format!("[tool error] {body}") };
    }
    Ok((body, is_error))
}

// --- PyToolProvider: the Python callback seam, as a Rust ToolProvider ----------

/// Wraps a Python `ToolProvider` object so the engine's `events_stream` can drive it.
/// `tool_schemas` reads `.tools_for(allow)[i].schema` under the GIL; `dispatch` builds
/// the `dispatch(...)` coroutine under the GIL, awaits it off-GIL (the
/// `pyo3_async_runtimes` bridge), and renders the result — a dispatch that raises is
/// caught and returned as `(ERROR: …, false)`, never propagated.
struct PyToolProvider {
    obj: Py<PyAny>,
}

#[async_trait::async_trait]
impl ToolProvider for PyToolProvider {
    fn tool_schemas(&self, allow: &[String]) -> Result<Vec<Value>, EngineError> {
        Python::with_gil(|py| -> PyResult<Vec<Value>> {
            let allow_list = pyo3::types::PyList::new(py, allow)?;
            let specs = self.obj.bind(py).call_method1("tools_for", (allow_list,))?;
            let mut schemas: Vec<Value> = Vec::new();
            for s in specs.try_iter()? {
                schemas.push(pythonize::depythonize(&s?.getattr("schema")?)?);
            }
            Ok(schemas)
        })
        .map_err(|e| EngineError::new(pyerr_brief(&e), "server_error", 500))
    }

    async fn dispatch(&self, name: &str, args: &Value) -> (String, bool) {
        let built = Python::with_gil(|py| -> PyResult<_> {
            let args_py = pythonize::pythonize(py, args)?;
            let awaitable = self.obj.bind(py).call_method1("dispatch", (name, args_py))?;
            pyo3_async_runtimes::tokio::into_future(awaitable)
        });
        let fut = match built {
            Ok(f) => f,
            Err(e) => return (format!("ERROR: {}", pyerr_brief(&e)), false),
        };
        match fut.await {
            Err(e) => (format!("ERROR: {}", pyerr_brief(&e)), false),
            Ok(tr) => match Python::with_gil(|py| render_tool_result_rs(tr.bind(py))) {
                Ok((body, is_error)) => (body, !is_error),
                Err(e) => (format!("ERROR: {}", pyerr_brief(&e)), false),
            },
        }
    }
}

// --- ModelRegistry pyclass (wraps engine::Registry) ---------------------------

/// An explicit, instance-scoped inferencer set — pass to `orchestrate`/
/// `orchestrate_events` via `registry=` to resolve config-file inferencers
/// (`ModelRegistry.from_config()`); omitting it uses the built-ins.
#[pyclass]
struct ModelRegistry {
    inner: engine::Registry,
}

#[pymethods]
impl ModelRegistry {
    #[new]
    fn new() -> Self {
        ModelRegistry { inner: engine::Registry::new() }
    }

    #[staticmethod]
    fn from_config() -> PyResult<ModelRegistry> {
        Ok(ModelRegistry { inner: engine::Registry::from_config().map_err(engine_err)? })
    }

    #[pyo3(signature = (name, base_url, *, api_key_env=None, extra_body=None))]
    fn add(
        &mut self,
        name: String,
        base_url: String,
        api_key_env: Option<String>,
        extra_body: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let extra_body = match extra_body {
            Some(b) => pythonize::depythonize(&b)?,
            None => json!({}),
        };
        self.inner.add(name, base_url, api_key_env, extra_body);
        Ok(())
    }

    fn get(&self, provider: &str) -> PyResult<Option<Py<PyAny>>> {
        match self.inner.get_json(provider) {
            Some(v) => Ok(Some(pyval(&v)?)),
            None => Ok(None),
        }
    }

    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn all(&self) -> PyResult<Py<PyAny>> {
        pyval(&self.inner.all_json())
    }
}

// --- stateless complete / complete_sync / provider_names ----------------------

/// Run one stateless turn against `<provider>/<model>` and return the assistant text
/// — an awaitable (so `await complete(...)` works for async embedders).
#[pyfunction]
#[pyo3(signature = (model, messages, *, options=None, params=None, api_key=None, base_url=None))]
fn complete<'py>(
    py: Python<'py>,
    model: String,
    messages: Bound<'py, PyAny>,
    options: Option<Bound<'py, PyAny>>,
    params: Option<Bound<'py, PyAny>>,
    api_key: Option<String>,
    base_url: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let msgs: Value = pythonize::depythonize(&messages)?;
    let req = engine::build_request(&model, msgs, depy(&options)?, depy(&params)?, api_key.as_deref(), base_url, false)
        .map_err(engine_err)?;
    future_into_py(py, async move { engine::run_complete(req).await.map_err(engine_err) })
}

/// Synchronous variant — for non-async embedders. HTTP runs off the GIL.
#[pyfunction]
#[pyo3(signature = (model, messages, *, options=None, params=None, api_key=None, base_url=None))]
fn complete_sync(
    py: Python<'_>,
    model: String,
    messages: Bound<'_, PyAny>,
    options: Option<Bound<'_, PyAny>>,
    params: Option<Bound<'_, PyAny>>,
    api_key: Option<String>,
    base_url: Option<String>,
) -> PyResult<String> {
    let msgs: Value = pythonize::depythonize(&messages)?;
    let req = engine::build_request(&model, msgs, depy(&options)?, depy(&params)?, api_key.as_deref(), base_url, false)
        .map_err(engine_err)?;
    py.allow_threads(move || engine::run_complete_blocking(req)).map_err(engine_err)
}

/// The built-in provider names (parity with `inferencers.names()`).
#[pyfunction]
fn provider_names() -> Vec<String> {
    engine::provider_names()
}

// --- orchestrate (the recipe↔tool loop) + orchestrate_events ------------------

/// One progress event from the loop as the OpenAI-ish event dict.
fn event_to_py(ev: &Event) -> PyResult<Py<PyAny>> {
    let v = match ev {
        Event::Delta(c) => json!({"type": "delta", "content": c}),
        Event::ToolCall { turn, name, args } => {
            json!({"type": "tool_call", "turn": turn, "name": name, "args": args})
        }
        Event::ToolResult { turn, name, ok } => {
            json!({"type": "tool_result", "turn": turn, "name": name, "ok": ok})
        }
        Event::Final(resp) => json!({"type": "final", "response": resp}),
    };
    pyval(&v)
}

/// Build the engine `Setup` from Python args (eager, raises on the call).
fn setup_from_py(
    recipe: &Bound<'_, PyAny>,
    user_msgs: &Bound<'_, PyAny>,
    tools: &Bound<'_, PyAny>,
    api_key: Option<String>,
    base_url: Option<String>,
    registry: Option<&ModelRegistry>,
) -> PyResult<engine::Setup> {
    let recipe_val: Value = pythonize::depythonize(recipe)?;
    let user_val: Value = pythonize::depythonize(user_msgs)?;
    let provider: Arc<dyn ToolProvider> = Arc::new(PyToolProvider { obj: tools.clone().unbind() });
    let reg = registry.map(|r| &r.inner);
    engine::build_setup(&recipe_val, &user_val, provider, api_key, base_url, reg).map_err(engine_err)
}

/// Run a recipe and return the final OpenAI response dict — the drainer over the
/// engine's `events_stream`.
#[pyfunction]
#[pyo3(signature = (recipe, user_msgs, tools, *, api_key=None, base_url=None, registry=None))]
fn orchestrate<'py>(
    py: Python<'py>,
    recipe: Bound<'py, PyAny>,
    user_msgs: Bound<'py, PyAny>,
    tools: Bound<'py, PyAny>,
    api_key: Option<String>,
    base_url: Option<String>,
    registry: Option<PyRef<'py, ModelRegistry>>,
) -> PyResult<Bound<'py, PyAny>> {
    let setup = setup_from_py(&recipe, &user_msgs, &tools, api_key, base_url, registry.as_deref())?;
    future_into_py(py, async move {
        let mut s = Box::pin(engine::events_stream(setup, false));
        while let Some(item) = s.next().await {
            match item {
                Ok(Event::Final(resp)) => {
                    return Python::with_gil(|py| Ok(pythonize::pythonize(py, &resp)?.unbind()))
                }
                Ok(_) => continue,
                Err(e) => return Err(engine_err(e)),
            }
        }
        Err(engine_err(EngineError::new(
            "orchestrate: loop ended without a final response",
            "server_error",
            500,
        )))
    })
}

/// An async iterator over the recipe loop's progress + `final` events.
#[pyclass]
struct EventIter {
    inner: Arc<Mutex<Pin<Box<dyn Stream<Item = Result<Event, EngineError>> + Send>>>>,
}

#[pymethods]
impl EventIter {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let mut s = inner.lock().await;
            match s.next().await {
                Some(Ok(ev)) => event_to_py(&ev),
                Some(Err(e)) => Err(engine_err(e)),
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

/// The recipe loop as an async iterator of progress + `final` events. With
/// `stream=True`, turns run over SSE and the iterator also yields `delta` events.
#[pyfunction]
#[pyo3(signature = (recipe, user_msgs, tools, *, api_key=None, base_url=None, registry=None, stream=false))]
fn orchestrate_events<'py>(
    _py: Python<'py>,
    recipe: Bound<'py, PyAny>,
    user_msgs: Bound<'py, PyAny>,
    tools: Bound<'py, PyAny>,
    api_key: Option<String>,
    base_url: Option<String>,
    registry: Option<PyRef<'py, ModelRegistry>>,
    stream: bool,
) -> PyResult<EventIter> {
    let setup = setup_from_py(&recipe, &user_msgs, &tools, api_key, base_url, registry.as_deref())?;
    Ok(EventIter { inner: Arc::new(Mutex::new(Box::pin(engine::events_stream(setup, stream)))) })
}

// --- streaming complete ------------------------------------------------------

/// An async iterator over assistant text deltas — `async for d in complete_stream(...)`.
#[pyclass]
struct DeltaStream {
    inner: Arc<Mutex<Pin<Box<dyn Stream<Item = Result<String, EngineError>> + Send>>>>,
}

#[pymethods]
impl DeltaStream {
    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let mut s = inner.lock().await;
            match s.next().await {
                Some(Ok(piece)) => Ok(piece),
                Some(Err(e)) => Err(engine_err(e)),
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

/// Stream one stateless turn as assistant text deltas (the inferencer's `/v1` SSE).
#[pyfunction]
#[pyo3(signature = (model, messages, *, options=None, params=None, api_key=None, base_url=None))]
fn complete_stream(
    model: String,
    messages: Bound<'_, PyAny>,
    options: Option<Bound<'_, PyAny>>,
    params: Option<Bound<'_, PyAny>>,
    api_key: Option<String>,
    base_url: Option<String>,
) -> PyResult<DeltaStream> {
    let msgs: Value = pythonize::depythonize(&messages)?;
    let req = engine::build_request(&model, msgs, depy(&options)?, depy(&params)?, api_key.as_deref(), base_url, true)
        .map_err(engine_err)?;
    Ok(DeltaStream { inner: Arc::new(Mutex::new(Box::pin(engine::complete_stream_events(req)))) })
}

// `module-name = "woollama.core"` (pyproject) → the init symbol must be `PyInit_core`,
// so this fn is named `core` (the last dotted component); `woollama` is the PEP 420 namespace.
#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<InferenceError>()?;
    m.py().get_type::<InferenceError>().setattr("__module__", "woollama.core")?;
    m.add_function(wrap_pyfunction!(complete, m)?)?;
    m.add_function(wrap_pyfunction!(complete_sync, m)?)?;
    m.add_function(wrap_pyfunction!(complete_stream, m)?)?;
    m.add_function(wrap_pyfunction!(provider_names, m)?)?;
    m.add_function(wrap_pyfunction!(orchestrate, m)?)?;
    m.add_function(wrap_pyfunction!(orchestrate_events, m)?)?;
    m.add_class::<EventIter>()?;
    m.add_class::<ModelRegistry>()?;
    m.add_class::<DeltaStream>()?;
    Ok(())
}
