//! `Run`: one execution of an agent, and the ways Python can read its events.
//!
//! The agent loop runs as a background Rust task and puts its events in a
//! queue. Python reads them in one of four ways:
//!
//! | Python                          | Each step gives                                   |
//! |---------------------------------|---------------------------------------------------|
//! | `async for event in run`        | the next event                                    |
//! | `async for events in run.batches()` | every event already waiting, as a list        |
//! | `async for text in run.ai_sdk()` | Vercel AI SDK stream text (`useChat`), see `ai_sdk` |
//! | `async for text in run.ag_ui(...)` | AG-UI stream text, see `ag_ui`                 |
//!
//! Each step is one hand-over from Rust to Python, and a hand-over costs far
//! more than the event itself (waking Python's event loop, taking the GIL).
//! So the last three never wait to fill a step, but take *everything* already
//! in the queue: with a fast model or a busy server, one hand-over carries many
//! events. With a slow model, each step still carries one event right away.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_async_runtimes::TaskLocals;
use serde_json::Value;
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinHandle;

use crate::ag_ui::AgUiEncoder;
use crate::agent::{RunInputs, run_in_background};
use crate::ai_sdk::AiSdkEncoder;
use crate::errors::AgentError;
use crate::events::AgentEvent;

/// Python: `async for event in run: ...` (or `run.batches()`, `run.ai_sdk()`, `run.ag_ui(...)`).
#[pyclass(frozen, module = "pyox_agents")]
pub struct Run {
    queue: Arc<EventQueue>,
}

impl Run {
    pub fn new(inputs: RunInputs, receiver: Receiver<PyResult<AgentEvent>>) -> Self {
        Run {
            queue: Arc::new(EventQueue {
                not_started: Mutex::new(Some(inputs)),
                receiver: tokio::sync::Mutex::new(receiver),
                background_task: Mutex::new(None),
            }),
        }
    }
}

#[pymethods]
impl Run {
    fn __aiter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.queue.start(py)?;
        let queue = self.queue.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match queue.next().await {
                Some(Ok(event)) => Python::attach(|py| event.into_python(py)),
                Some(Err(error)) => Err(error),
                None => Err(PyStopAsyncIteration::new_err(())), // the run is over
            }
        })
    }

    /// Python: `async for events in run.batches()`: lists of events, each
    /// holding every event that was waiting (at least one).
    fn batches(&self) -> EventBatches {
        EventBatches {
            queue: self.queue.clone(),
            error_after_batch: Mutex::new(None),
        }
    }

    /// Python: `StreamingResponse(run.ai_sdk(), headers=ai_sdk.HEADERS)`: the
    /// run as a Vercel AI SDK UI message stream, as text.
    fn ai_sdk(&self) -> ProtocolStream {
        ProtocolStream::new(self.queue.clone(), Encoder::AiSdk(AiSdkEncoder::default()))
    }

    /// Python: `StreamingResponse(run.ag_ui(thread_id, run_id), headers=ag_ui.HEADERS)`:
    /// the run as an AG-UI event stream, as text.
    fn ag_ui(&self, thread_id: String, run_id: String) -> ProtocolStream {
        ProtocolStream::new(self.queue.clone(), Encoder::AgUi(AgUiEncoder::new(thread_id, run_id)))
    }
}

/// The run's event queue, shared by `Run` and the iterators made from it.
struct EventQueue {
    /// The run's inputs until the first read starts it.
    not_started: Mutex<Option<RunInputs>>,
    /// `tokio::sync::Mutex`: a lock that can be held while waiting (`.await`).
    receiver: tokio::sync::Mutex<Receiver<PyResult<AgentEvent>>>,
    background_task: Mutex<Option<JoinHandle<()>>>,
}

impl EventQueue {
    /// Starts the agent loop, on the first read. It must happen inside Python's
    /// running event loop, which Python tools run on.
    fn start(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(inputs) = self.not_started.lock().unwrap().take() {
            let python_task_locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
            let task = pyo3_async_runtimes::tokio::get_runtime().spawn(run_in_background(inputs, python_task_locals));
            *self.background_task.lock().unwrap() = Some(task);
        }
        Ok(())
    }

    /// The next event, or `None` when the run is over.
    async fn next(&self) -> Option<PyResult<AgentEvent>> {
        self.receiver.lock().await.recv().await
    }

    /// Waits for one event, then takes every other event already waiting
    /// (without waiting for more). Empty when the run is over.
    async fn next_batch(&self) -> Vec<PyResult<AgentEvent>> {
        let mut receiver = self.receiver.lock().await;
        let Some(first) = receiver.recv().await else {
            return Vec::new();
        };
        let mut batch = vec![first];
        while let Ok(event) = receiver.try_recv() {
            batch.push(event);
        }
        batch
    }
}

impl Drop for EventQueue {
    /// When Python stops reading early (`break`, or a client that disconnects),
    /// the run and its iterators are dropped: stop the agent loop too.
    fn drop(&mut self) {
        if let Some(task) = self.background_task.lock().unwrap().take() {
            task.abort();
        }
    }
}

/// Python: `async for events in run.batches()`.
#[pyclass(frozen, module = "pyox_agents")]
pub struct EventBatches {
    queue: Arc<EventQueue>,
    /// A run error that came after some events: raised once those are read.
    error_after_batch: Mutex<Option<PyErr>>,
}

#[pymethods]
impl EventBatches {
    fn __aiter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __anext__<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = this.py();
        if let Some(error) = this.get().error_after_batch.lock().unwrap().take() {
            return Err(error);
        }
        this.get().queue.start(py)?;
        let batches = this.unbind();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let batch = batches.get().queue.next_batch().await;
            if batch.is_empty() {
                return Err(PyStopAsyncIteration::new_err(()));
            }
            Python::attach(|py| {
                let events = PyList::empty(py);
                for item in batch {
                    match item {
                        Ok(event) => events.append(event.into_python(py)?)?,
                        // An error is always the run's last item.
                        Err(error) if events.is_empty() => return Err(error),
                        Err(error) => *batches.get().error_after_batch.lock().unwrap() = Some(error),
                    }
                }
                Ok(events.unbind())
            })
        })
    }
}

/// Python: `async for text in run.ai_sdk()` (or `run.ag_ui(...)`): ready
/// server-sent events text for a web response.
#[pyclass(frozen, module = "pyox_agents")]
pub struct ProtocolStream {
    queue: Arc<EventQueue>,
    state: Arc<Mutex<StreamState>>,
}

struct StreamState {
    encoder: Encoder,
    started: bool,
    finished: bool,
}

impl StreamState {
    /// The protocol text for a batch of run events (the stream's start first).
    fn encode(&mut self, batch: Vec<PyResult<AgentEvent>>) -> PyResult<String> {
        let mut text = String::new();
        if !self.started {
            self.encoder.start(&mut text);
            self.started = true;
        }
        if batch.is_empty() {
            self.finished = true; // the run ended without a last event (it was stopped)
        }
        for item in batch {
            match item {
                Ok(event) => {
                    self.finished = matches!(event, AgentEvent::RunFinished(_));
                    self.encoder.add_event(&event, &mut text);
                }
                // A pyox-agents error (API down, max turns, ...) becomes the
                // protocol's error event. Anything else is a bug: raise it.
                Err(error) => {
                    let (message, code) = Python::attach(|py| {
                        if !error.is_instance_of::<AgentError>(py) {
                            return Err(error.clone_ref(py));
                        }
                        let code = error.get_type(py).name()?.to_string();
                        Ok((error.value(py).to_string(), code))
                    })?;
                    self.encoder.add_error(&message, &code, &mut text);
                    self.finished = true;
                }
            }
        }
        Ok(text)
    }
}

impl ProtocolStream {
    fn new(queue: Arc<EventQueue>, encoder: Encoder) -> Self {
        let state = StreamState {
            encoder,
            started: false,
            finished: false,
        };
        ProtocolStream {
            queue,
            state: Arc::new(Mutex::new(state)),
        }
    }
}

#[pymethods]
impl ProtocolStream {
    fn __aiter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.queue.start(py)?;
        let queue = self.queue.clone();
        let state = self.state.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            loop {
                if state.lock().unwrap().finished {
                    return Err(PyStopAsyncIteration::new_err(()));
                }
                let batch = queue.next_batch().await;
                let text = state.lock().unwrap().encode(batch)?;
                // Some events have no text of their own (AG-UI lists waiting calls at the end).
                if !text.is_empty() {
                    return Ok(text);
                }
            }
        })
    }
}

/// The protocols a run can be streamed as.
enum Encoder {
    AiSdk(AiSdkEncoder),
    AgUi(AgUiEncoder),
}

impl Encoder {
    fn start(&mut self, out: &mut String) {
        match self {
            Encoder::AiSdk(encoder) => encoder.start(out),
            Encoder::AgUi(encoder) => encoder.start(out),
        }
    }

    fn add_event(&mut self, event: &AgentEvent, out: &mut String) {
        match self {
            Encoder::AiSdk(encoder) => encoder.add_event(event, out),
            Encoder::AgUi(encoder) => encoder.add_event(event, out),
        }
    }

    fn add_error(&mut self, message: &str, code: &str, out: &mut String) {
        match self {
            Encoder::AiSdk(encoder) => encoder.add_error(message, out),
            Encoder::AgUi(encoder) => encoder.add_error(message, code, out),
        }
    }
}

/// Adds one server-sent event to `out`: `data: {...json...}` and a blank line.
pub fn write_sse(out: &mut String, data: &Value) {
    out.push_str("data: ");
    out.push_str(&data.to_string());
    out.push_str("\n\n");
}

/// A new id like `msg_18f2a9c3e4b5_7`: unique in this process, and between
/// restarts (it starts with the time).
pub fn new_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|time| time.as_nanos())
        .unwrap_or_default();
    format!("{prefix}_{nanos:x}_{count}")
}
