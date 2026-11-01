//! Telemetry: OpenTelemetry spans for runs, model requests and tool calls.
//!
//! Rust knows nothing about OpenTelemetry. When tracing is on, Python gives
//! the run a small "recorder" (`_Recorder` in `pyox_agents/telemetry.py`), and
//! the agent loop tells it what happened, with times measured here. The
//! recorder makes the spans with the app's TracerProvider, so the app's
//! exporters, sampler and vendor SDK (Logfire, Langfuse...) see them.
//!
//! - Run and tool call spans are *live*: they are open while the run or the
//!   tool works, and the Python calls inside them (hooks, tools, `on_request`)
//!   run in a copy of the contextvars where that span is current. So the
//!   app's own spans inside a tool nest under its `execute_tool` span.
//! - A model request span is sent in one call, after the request: no Python
//!   code runs inside a model request.
//!
//! When tracing is off there is no recorder, and every function here returns
//! at once without touching Python.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_async_runtimes::TaskLocals;
use pyox_tools::usage::Usage;
use serde_json::{Value, json};

use crate::events::RunFinished;
use crate::json_convert::json_to_python;
use crate::messages::{Message, ToolCall};
use crate::model_api::{AssistantTurn, ModelRequest};

/// Nanoseconds since 1970, like Python's `time.time_ns()`: the time format
/// OpenTelemetry takes for span starts and ends.
pub fn now_ns() -> u64 {
    let since_1970 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    since_1970.as_nanos() as u64
}

/// Where a run's `invoke_agent` span goes. Cheap to clone.
#[derive(Clone, Default)]
pub enum Telemetry {
    /// Tracing is off (the default): no spans, no Python.
    #[default]
    Off,
    /// A run started with `execute()`: the recorder Python made for it.
    /// (`Arc`, a shared pointer, so clones need no Python.)
    Recorder(Arc<Py<PyAny>>),
    /// A subagent's run: its span goes under the tool call that runs it.
    InTool(Span),
}

impl Telemetry {
    /// Opens the run's `invoke_agent` span. Returns it, and the TaskLocals
    /// the run's Python calls use: `python_task_locals` with the run's span
    /// as the current span.
    pub async fn start_run(
        &self,
        name: Option<&str>,
        model: &str,
        base_url: &str,
        conversation_id: Option<&str>,
        python_task_locals: &TaskLocals,
    ) -> (Span, TaskLocals) {
        // Both the recorder and a tool call's span have a `start_run` method.
        let opener = match self {
            Telemetry::Off => None,
            Telemetry::Recorder(recorder) => Some(recorder.clone()),
            Telemetry::InTool(tool_span) => tool_span.0.as_ref().map(|span| span.handle.clone()),
        };
        let Some(opener) = opener else {
            return (Span::default(), python_task_locals.clone());
        };
        let fields = json!({
            "start_ns": now_ns(),
            "name": name,
            "model": model,
            "base_url": base_url,
            "conversation_id": conversation_id,
        });
        open_span(opener, "start_run", fields, python_task_locals).await
    }
}

/// A live span (`invoke_agent` or `execute_tool`), or nothing when tracing
/// is off. Clones share the same span.
#[derive(Clone, Default)]
pub struct Span(Option<Arc<OpenSpan>>);

struct OpenSpan {
    /// The Python object for the span (`_RunSpan` or `_ToolSpan` in telemetry.py).
    handle: Arc<Py<PyAny>>,
    /// Copy prompts, answers, tool arguments and results into the spans.
    /// Rust only builds that data when this is on.
    capture_content: bool,
    /// Set once the span is finished.
    ended: AtomicBool,
}

impl Drop for OpenSpan {
    /// A span dropped before it was finished belongs to a run that was
    /// dropped (Python stopped reading it, so the agent loop was aborted).
    /// End it as cancelled, so no span stays open forever. `try_attach` does
    /// nothing when Python is shutting down.
    fn drop(&mut self) {
        if self.ended.load(Ordering::Relaxed) {
            return;
        }
        let end_ns = now_ns();
        Python::try_attach(|py| {
            if let Err(error) = self.handle.call_method1(py, "cancel", (end_ns,)) {
                error.write_unraisable(py, None);
            }
        });
    }
}

/// What one model request did, for its `chat` span.
pub struct ModelRequestRecord<'a> {
    pub turn: u32,
    pub started_at: u64,
    pub ended_at: u64,
    /// The request as sent (after the `before_model_request` hooks).
    pub request: &'a ModelRequest<'a>,
    /// The answer, or the error. `None` when `run.stop()` cut the answer short.
    pub outcome: Option<&'a PyResult<AssistantTurn>>,
}

impl Span {
    /// Sends the `chat` span of one model request, under this run's span.
    pub async fn record_model_request(&self, record: ModelRequestRecord<'_>) {
        let Some(span) = &self.0 else { return };
        let request = record.request;
        let mut fields = json!({
            "turn": record.turn,
            "start_ns": record.started_at,
            "end_ns": record.ended_at,
            "model": request.model,
            "model_parameters": request.model_parameters,
        });
        let mut error = None;
        match record.outcome {
            Some(Ok(turn)) => {
                fields["first_chunk_ns"] = json!(turn.first_chunk_at);
                fields["stop_reason"] = json!(turn.stop_reason.name());
                fields["usage"] = json!(turn.usage);
                fields["response_id"] = json!(turn.response_id);
                fields["request_id"] = json!(turn.request_id);
                fields["retries"] = json!(turn.retries);
            }
            Some(Err(failure)) => error = Some(exception_of(failure)),
            None => fields["stop_reason"] = json!("stopped"),
        }
        if span.capture_content {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|tool| json!({"name": tool.name, "description": tool.description, "parameters": tool.parameters}))
                .collect();
            fields["instructions"] = json!(request.instructions);
            fields["messages"] = json!(request.messages);
            fields["tools"] = json!(tools);
            if let Some(Ok(turn)) = record.outcome {
                fields["output"] = json!({"text": turn.text, "tool_calls": turn.tool_calls});
            }
        }
        let handle = span.handle.clone();
        in_python(move |py| {
            let arguments = keyword_arguments(py, &fields)?;
            arguments.set_item("error", error)?;
            handle.call_method(py, "model_request", (), Some(&arguments))
        })
        .await;
    }

    /// Opens the `execute_tool` span of one call, under this run's span.
    /// Returns it, and the TaskLocals the call's Python code (hooks, the
    /// approval function, the tool itself) runs with.
    pub async fn start_tool(
        &self,
        call: &ToolCall,
        description: &str,
        kind: &str,
        python_task_locals: &TaskLocals,
    ) -> (Span, TaskLocals) {
        let Some(span) = &self.0 else {
            return (Span::default(), python_task_locals.clone());
        };
        let mut fields = json!({
            "start_ns": now_ns(),
            "call_id": call.id,
            "name": call.name,
            "description": description,
            "kind": kind,
        });
        if span.capture_content {
            fields["arguments"] = call.arguments.clone();
        }
        open_span(span.handle.clone(), "start_tool", fields, python_task_locals).await
    }

    /// Ends a tool call's span with what the call gave: its tool message, or the
    /// error that stopped the run.
    pub async fn finish_tool(&self, outcome: &PyResult<(Message, Usage)>) {
        let mut fields = json!({"end_ns": now_ns()});
        let mut error = None;
        match outcome {
            Ok((message @ Message::Tool { content, is_error, .. }, _)) => {
                fields["is_error"] = json!(is_error);
                fields["waiting"] = json!(message.is_waiting());
                if self.capture_content() {
                    fields["result"] = json!(content.text());
                }
            }
            Ok(_) => unreachable!("a tool call always gives a tool message"),
            Err(failure) => error = Some(exception_of(failure)),
        }
        self.finish(fields, error).await;
    }

    /// Ends a run's span with how the run ended. `usage` is the run's total.
    pub async fn finish_run(&self, outcome: &PyResult<RunFinished>, usage: Usage) {
        let mut fields = json!({"end_ns": now_ns(), "usage": usage});
        let mut error = None;
        match outcome {
            Ok(finished) => {
                fields["turns"] = json!(finished.turns);
                fields["stop_reason"] = json!(finished.stop_reason.name());
                fields["pending"] = json!(finished.pending.len());
            }
            Err(failure) => error = Some(exception_of(failure)),
        }
        self.finish(fields, error).await;
    }

    async fn finish(&self, fields: Value, error: Option<Py<PyAny>>) {
        let Some(span) = self.0.clone() else { return };
        span.ended.store(true, Ordering::Relaxed);
        in_python(move |py| {
            let arguments = keyword_arguments(py, &fields)?;
            arguments.set_item("error", error)?;
            span.handle.call_method(py, "finish", (), Some(&arguments))
        })
        .await;
    }

    /// A moment in a tool call: `"input_requested"`, `"input_replied"` or
    /// `"state_changed"`, with its details (`telemetry.py` names them).
    pub async fn add_event(&self, name: &'static str, attributes: Value) {
        let Some(span) = &self.0 else { return };
        let (handle, time_ns) = (span.handle.clone(), now_ns());
        in_python(move |py| {
            let attributes = json_to_python(py, &attributes)?;
            handle.call_method1(py, "add_event", (name, attributes, time_ns))
        })
        .await;
    }

    pub fn capture_content(&self) -> bool {
        self.0.as_ref().is_some_and(|span| span.capture_content)
    }
}

/// Calls `opener.method(**fields, run_context=...)`, which opens a span and
/// returns `(span, context)`: the Python span object, and a copy of the
/// contextvars of `python_task_locals` where that span is current.
async fn open_span(
    opener: Arc<Py<PyAny>>,
    method: &'static str,
    fields: Value,
    python_task_locals: &TaskLocals,
) -> (Span, TaskLocals) {
    let locals = python_task_locals.clone();
    let opened = in_python(move |py| {
        let arguments = keyword_arguments(py, &fields)?;
        arguments.set_item("run_context", locals.context(py))?;
        let opened = opener.call_method(py, method, (), Some(&arguments))?;
        let (handle, context): (Py<PyAny>, Bound<'_, PyAny>) = opened.bind(py).extract()?;
        let capture_content = handle.getattr(py, "capture_content")?.extract(py)?;
        let span = OpenSpan {
            handle: Arc::new(handle),
            capture_content,
            ended: AtomicBool::new(false),
        };
        Ok((Span(Some(Arc::new(span))), locals.with_context(context)))
    })
    .await;
    // The recorder failed: the run goes on without this span.
    opened.unwrap_or_else(|| (Span::default(), python_task_locals.clone()))
}

/// Runs `work` with Python, on a thread meant for blocking work (like Python
/// tools and hooks): a slow span processor must not stall the network tasks.
///
/// Telemetry never breaks a run: an error goes to `sys.unraisablehook`
/// (Python prints it) and gives `None`.
async fn in_python<T: Send + 'static>(
    work: impl for<'py> FnOnce(Python<'py>) -> PyResult<T> + Send + 'static,
) -> Option<T> {
    let outcome = tokio::task::spawn_blocking(move || {
        Python::attach(|py| match work(py) {
            Ok(value) => Some(value),
            Err(error) => {
                error.write_unraisable(py, None);
                None
            }
        })
    });
    outcome.await.ok().flatten()
}

/// `fields` (a JSON object) as Python keyword arguments.
fn keyword_arguments<'py>(py: Python<'py>, fields: &Value) -> PyResult<Bound<'py, PyDict>> {
    Ok(json_to_python(py, fields)?.cast_into::<PyDict>()?)
}

/// The Python exception object of `error`, for the recorder to read its type and fields.
fn exception_of(error: &PyErr) -> Py<PyAny> {
    Python::attach(|py| error.value(py).clone().into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyModule;

    /// A recorder that only lists what it was asked (a real Python object, not a mock).
    const RECORDER: &std::ffi::CStr = c"
import contextvars

calls = []

class Span:
    capture_content = False

    def finish(self, **fields):
        calls.append('finish')

    def cancel(self, end_ns):
        calls.append('cancel')

class Recorder:
    def start_run(self, run_context, **fields):
        calls.append('start_run')
        return Span(), contextvars.copy_context()
";

    #[tokio::test]
    async fn a_span_dropped_before_it_finished_ends_as_cancelled() {
        Python::initialize();
        let (telemetry, python_task_locals, calls) = Python::attach(|py| {
            let module = PyModule::from_code(py, RECORDER, c"recorder.py", c"recorder").unwrap();
            let recorder = module.getattr("Recorder").unwrap().call0().unwrap().unbind();
            let context = py.import("contextvars").unwrap().call_method0("copy_context").unwrap();
            let locals = TaskLocals::new(py.None().into_bound(py)).with_context(context);
            (
                Telemetry::Recorder(Arc::new(recorder)),
                locals,
                module.getattr("calls").unwrap().unbind(),
            )
        });
        let calls = || Python::attach(|py| calls.extract::<Vec<String>>(py).unwrap());

        let (finished, _) = telemetry.start_run(None, "m", "", None, &python_task_locals).await;
        finished.finish(json!({}), None).await;
        drop(finished);
        assert_eq!(calls(), ["start_run", "finish"]);

        let (dropped, _) = telemetry.start_run(None, "m", "", None, &python_task_locals).await;
        drop(dropped); // like a run aborted in the middle
        assert_eq!(calls(), ["start_run", "finish", "start_run", "cancel"]);
    }

    #[tokio::test]
    async fn without_a_recorder_nothing_happens() {
        let span = Span::default();
        assert!(!span.capture_content());
        // No Python here: none of these may need it.
        span.add_event("state_changed", json!({"delta": []})).await;
        span.finish(json!({}), None).await;
    }
}
