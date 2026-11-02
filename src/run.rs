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
//! A `SyncRun` (`agent.execute_sync(...)`) is read the same four ways with a
//! plain `for`. The same agent loop runs on the same Rust runtime; the reading
//! thread waits with the GIL released, and meanwhile runs the run's sync
//! Python calls (see `python_call`). No Python event loop is needed.
//!
//! Each step is one hand-over from Rust to Python, and a hand-over costs far
//! more than the event itself (waking Python's event loop, taking the GIL).
//! So the last three never wait to fill a step, but take *everything* already
//! in the queue: with a fast model or a busy server, one hand-over carries many
//! events. With a slow model, each step still carries one event right away.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use std::future::Future;

use pyo3::exceptions::{PyStopAsyncIteration, PyStopIteration, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde_json::Value;
use tokio::sync::mpsc::Receiver;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::ag_ui::AgUiEncoder;
use crate::agent_loop::{RunInputs, run_in_background};
use crate::ai_sdk::AiSdkEncoder;
use crate::errors::AgentError;
use crate::events::AgentEvent;
use crate::python_call::{CallerThread, PythonCalls, check_not_forked, wait};
use crate::python_exit::future_into_py;

/// Python: `async for event in run: ...` (or `run.batches()`, `run.ai_sdk()`, `run.ag_ui(...)`).
///
/// `generic`, like the classes below: Python can write `Run[Ticket]`, the type of the final `output`.
#[pyclass(frozen, generic, module = "pyox_agents")]
pub struct Run {
    queue: Arc<EventQueue>,
    /// Sets the run's `StopSignal` (see `stop`).
    stop_sender: watch::Sender<bool>,
}

/// Python: `for event in run: ...` (or `run.batches()`, `run.ai_sdk()`, `run.ag_ui(...)`),
/// from `agent.execute_sync(...)`.
#[pyclass(frozen, generic, module = "pyox_agents")]
pub struct SyncRun {
    queue: Arc<EventQueue>,
    stop_sender: watch::Sender<bool>,
}

/// A new run of `inputs`: a `Run`, or a `SyncRun` when `sync` is set.
pub fn new_run(
    py: Python<'_>,
    inputs: RunInputs,
    receiver: Receiver<PyResult<AgentEvent>>,
    stop_sender: watch::Sender<bool>,
    sync: bool,
) -> PyResult<Py<PyAny>> {
    let queue = Arc::new(EventQueue {
        not_started: Mutex::new(Some(inputs)),
        receiver: tokio::sync::Mutex::new(Some(receiver)),
        background_task: Mutex::new(None),
        caller_thread: sync.then(CallerThread::new),
    });
    if sync {
        Ok(Py::new(py, SyncRun { queue, stop_sender })?.into_any())
    } else {
        Ok(Py::new(py, Run { queue, stop_sender })?.into_any())
    }
}

#[pymethods]
impl Run {
    fn __aiter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.queue.start_async(py)?;
        let queue = self.queue.clone();
        future_into_py(py, async move {
            match queue.next().await {
                Some(Ok(event)) => Python::attach(|py| event.into_python(py)),
                Some(Err(error)) => Err(error),
                None => Err(PyStopAsyncIteration::new_err(())), // the run is over
            }
        })
    }

    /// Python: `run.stop()`. Ends the run at the next safe point, keeping what
    /// it did: an answer being streamed ends at once (its text so far is
    /// kept), tools already running finish, and no new request is sent.
    /// Keep reading: the run still ends with `RunFinished(stop_reason="stopped")`.
    /// (Stopping reading instead, with `break`, aborts the run and keeps nothing.)
    fn stop(&self) {
        // `send_replace` sets the value even when nobody waits for it yet.
        self.stop_sender.send_replace(true);
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

#[pymethods]
impl SyncRun {
    fn __iter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.queue.wait(py, self.queue.next())? {
            Some(Ok(event)) => event.into_python(py),
            Some(Err(error)) => Err(error),
            None => Err(PyStopIteration::new_err(())), // the run is over
        }
    }

    fn __enter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    /// Leaving `with agent.execute_sync(...) as run:` closes the run.
    fn __exit__(&self, _type: Py<PyAny>, _value: Py<PyAny>, _traceback: Py<PyAny>) {
        self.queue.abort();
    }

    /// Python: `run.close()`. Ends the run at once and keeps nothing, like
    /// `break` (use `stop()` to keep what it did).
    fn close(&self) {
        self.queue.abort();
    }

    /// Python: `run.stop()`, from any thread: like `Run.stop()`.
    fn stop(&self) {
        self.stop_sender.send_replace(true);
    }

    /// Python: `for events in run.batches()`, like `Run.batches()`.
    fn batches(&self) -> EventBatches {
        EventBatches {
            queue: self.queue.clone(),
            error_after_batch: Mutex::new(None),
        }
    }

    /// Python: `StreamingHttpResponse(run.ai_sdk(), headers=ai_sdk.HEADERS)` in Django.
    fn ai_sdk(&self) -> ProtocolStream {
        ProtocolStream::new(self.queue.clone(), Encoder::AiSdk(AiSdkEncoder::default()))
    }

    /// Python: `StreamingHttpResponse(run.ag_ui(thread_id, run_id), headers=ag_ui.HEADERS)`.
    fn ag_ui(&self, thread_id: String, run_id: String) -> ProtocolStream {
        ProtocolStream::new(self.queue.clone(), Encoder::AgUi(AgUiEncoder::new(thread_id, run_id)))
    }
}

/// The run's event queue, shared by `Run` and the iterators made from it.
struct EventQueue {
    /// The run's inputs until the first read starts it.
    not_started: Mutex<Option<RunInputs>>,
    /// `tokio::sync::Mutex`: a lock that can be held while waiting (`.await`).
    /// `None` once the run's last item was read (see `is_last`).
    receiver: tokio::sync::Mutex<Option<Receiver<PyResult<AgentEvent>>>>,
    background_task: Mutex<Option<JoinHandle<()>>>,
    /// Sync runs only: the thread reading the run, which runs its sync Python calls.
    caller_thread: Option<CallerThread>,
}

impl EventQueue {
    /// Starts the agent loop, on the first read. An async run must start inside
    /// Python's running event loop, which its async Python functions run on.
    fn start(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(inputs) = self.not_started.lock().unwrap().take() {
            check_not_forked()?;
            let python_calls = match &self.caller_thread {
                Some(caller_thread) => PythonCalls::for_sync_run(py, caller_thread.clone())?,
                None => PythonCalls::for_async_run(py)?,
            };
            let task = pyo3_async_runtimes::tokio::get_runtime().spawn(run_in_background(inputs, python_calls));
            *self.background_task.lock().unwrap() = Some(task);
        }
        Ok(())
    }

    /// For the async readers (`async for`): starts the run.
    fn start_async(&self, py: Python<'_>) -> PyResult<()> {
        if self.caller_thread.is_some() {
            return Err(PyTypeError::new_err(
                "A run from execute_sync() is read with `for`, not `async for`",
            ));
        }
        self.start(py)
    }

    /// For the sync readers (`for`): starts the run, and waits here for `future`
    /// (the next event or batch), running the run's sync Python calls meanwhile.
    /// An error while waiting (Ctrl+C) ends the run.
    fn wait<T: Send>(&self, py: Python<'_>, future: impl Future<Output = T> + Send) -> PyResult<T> {
        let Some(caller_thread) = &self.caller_thread else {
            return Err(PyTypeError::new_err(
                "A run from execute() is read with `async for`; for `for`, use execute_sync()",
            ));
        };
        self.start(py)?;
        wait(py, Some(caller_thread), future).inspect_err(|_| self.abort())
    }

    /// Ends the run at once (Python stopped reading, or closed it).
    fn abort(&self) {
        if let Some(task) = self.background_task.lock().unwrap().take() {
            task.abort();
        }
    }

    /// The next event, or `None` when the run is over.
    async fn next(&self) -> Option<PyResult<AgentEvent>> {
        let mut slot = self.receiver.lock().await;
        let item = slot.as_mut()?.recv().await;
        if item.as_ref().is_none_or(is_last) {
            *slot = None;
        }
        item
    }

    /// Waits for one event, then takes every other event already waiting
    /// (without waiting for more). Empty when the run is over.
    async fn next_batch(&self) -> Vec<PyResult<AgentEvent>> {
        let mut slot = self.receiver.lock().await;
        let Some(receiver) = slot.as_mut() else {
            return Vec::new();
        };
        let Some(first) = receiver.recv().await else {
            *slot = None;
            return Vec::new();
        };
        let mut over = is_last(&first);
        let mut batch = vec![first];
        while !over && let Ok(event) = receiver.try_recv() {
            over = is_last(&event);
            batch.push(event);
        }
        if over {
            *slot = None;
        }
        batch
    }
}

/// The run's last item: its `RunFinished`, or the error that ended it.
///
/// The queue ends there, and drops its receiving end: whatever is sent after
/// is thrown away. Waiting for every sender to be gone instead would hang
/// the stream while anything still holds one, such as a tool's `ctx` kept in
/// a cache or in a library's background task.
fn is_last(item: &PyResult<AgentEvent>) -> bool {
    matches!(item, Ok(AgentEvent::RunFinished(_)) | Err(_))
}

impl Drop for EventQueue {
    /// When Python stops reading early (`break`, or a client that disconnects),
    /// the run and its iterators are dropped: stop the agent loop too.
    fn drop(&mut self) {
        self.abort();
    }
}

/// Python: `async for events in run.batches()`.
#[pyclass(frozen, generic, module = "pyox_agents")]
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
        this.get().queue.start_async(py)?;
        let batches = this.unbind();
        future_into_py(py, async move {
            let batch = batches.get().queue.next_batch().await;
            if batch.is_empty() {
                return Err(PyStopAsyncIteration::new_err(()));
            }
            Python::attach(|py| batches.get().to_python(py, batch))
        })
    }

    fn __iter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        if let Some(error) = self.error_after_batch.lock().unwrap().take() {
            return Err(error);
        }
        let batch = self.queue.wait(py, self.queue.next_batch())?;
        if batch.is_empty() {
            return Err(PyStopIteration::new_err(()));
        }
        self.to_python(py, batch)
    }
}

impl EventBatches {
    /// The batch as a Python list of events. An error after some events is
    /// kept, and raised by the next read.
    fn to_python(&self, py: Python<'_>, batch: Vec<PyResult<AgentEvent>>) -> PyResult<Py<PyList>> {
        let events = PyList::empty(py);
        for item in batch {
            match item {
                Ok(event) => events.append(event.into_python(py)?)?,
                // An error is always the run's last item.
                Err(error) if events.is_empty() => return Err(error),
                Err(error) => *self.error_after_batch.lock().unwrap() = Some(error),
            }
        }
        Ok(events.unbind())
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
        if !self.finished {
            self.encoder.end_batch(&mut text);
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
        self.queue.start_async(py)?;
        let queue = self.queue.clone();
        let state = self.state.clone();
        future_into_py(py, async move {
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

    fn __iter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<String> {
        loop {
            if self.state.lock().unwrap().finished {
                return Err(PyStopIteration::new_err(()));
            }
            let batch = self.queue.wait(py, self.queue.next_batch())?;
            let text = self.state.lock().unwrap().encode(batch)?;
            if !text.is_empty() {
                return Ok(text);
            }
        }
    }

    /// Ends the run at once. Django's `StreamingHttpResponse` calls it when
    /// the response is closed (for example, the client went away).
    fn close(&self) {
        self.queue.abort();
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

    /// Called after each batch of events: what was held back to send once per batch.
    fn end_batch(&mut self, out: &mut String) {
        if let Encoder::AiSdk(encoder) = self {
            encoder.end_batch(out);
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
