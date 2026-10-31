//! The events Python receives while iterating over a run.
//!
//! Each event is a Rust struct exposed as a read-only Python class:
//! - `#[pyclass(frozen)]` makes the class immutable from Python.
//! - `get_all` creates a Python property for every field.
//!
//! Fields stay in Rust until Python reads them, and there is no per-object
//! `__dict__`, so events are small and cheap to create.
//!
//! A subagent's events come in the same stream as the main agent's. Their
//! `subagent_id` says which subagent sent them (`None` for the main agent),
//! and their call ids start with it (`call_0/call_1`), so ids stay unique.

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyox_tools::usage::Usage;
use serde_json::Value;
use tokio::sync::mpsc::Sender;

use crate::human_input::InputRequested;
use crate::json_convert::json_to_python;
use crate::messages::Message;

/// A piece of the assistant's answer, streamed as soon as it arrives.
#[pyclass(frozen, get_all, module = "pyox_agents")]
pub struct TextDelta {
    pub text: String,
    /// The subagent writing it; `None` for the main agent.
    pub subagent_id: Option<String>,
}

/// The model asked to call a tool and the tool is about to run.
#[pyclass(frozen, module = "pyox_agents")]
pub struct ToolStarted {
    #[pyo3(get)]
    pub call_id: String,
    #[pyo3(get)]
    pub name: String,
    pub arguments: Value,
    #[pyo3(get)]
    pub subagent_id: Option<String>,
}

/// A tool finished. `is_error` is true if it raised an exception.
#[pyclass(frozen, module = "pyox_agents")]
pub struct ToolFinished {
    #[pyo3(get)]
    pub call_id: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub result: String,
    #[pyo3(get)]
    pub is_error: bool,
    /// Data for the app from `ToolReturn(metadata=...)`; the model never sees it.
    pub metadata: Value,
    /// The app gave this result (a frontend tool), so protocol streams don't send it back.
    pub from_app: bool,
    #[pyo3(get)]
    pub subagent_id: Option<String>,
}

/// A tool changed the run's state (`ctx.state`).
#[pyclass(frozen, module = "pyox_agents")]
pub struct StateChanged {
    /// The whole state, as it is now.
    pub state: Value,
    /// What changed, as a JSON Patch (RFC 6902): `[{"op": "replace", "path": "/count", "value": 2}]`.
    pub delta: Value,
    #[pyo3(get)]
    pub subagent_id: Option<String>,
}

/// A tool call started a subagent (see `subagents`). Its events follow, with
/// `subagent_id` set to this `call_id`.
#[pyclass(frozen, get_all, module = "pyox_agents")]
pub struct SubagentStarted {
    /// The id of the tool call running the subagent, which is also the subagent's id.
    pub call_id: String,
    /// The tool's name.
    pub name: String,
    /// What the model asked the subagent to do.
    pub task: String,
    /// The agent that started it: `None` for the main agent.
    pub subagent_id: Option<String>,
}

/// A subagent stopped: it finished, failed, or waits for a person.
#[pyclass(frozen, get_all, module = "pyox_agents")]
pub struct SubagentFinished {
    pub call_id: String,
    pub name: String,
    /// What the calling agent reads as the tool's result (empty while it waits).
    pub result: String,
    pub is_error: bool,
    /// It waits for a person: its requests are in `RunFinished.pending`.
    pub waiting: bool,
    /// Tokens the subagent used (also counted in the run's usage).
    pub usage: Usage,
    pub subagent_id: Option<String>,
}

/// The last event of a run.
#[derive(Clone)]
#[pyclass(frozen, module = "pyox_agents")]
pub struct RunFinished {
    /// Text of the final assistant message.
    #[pyo3(get)]
    pub text: String,
    /// Number of model requests made during the run.
    #[pyo3(get)]
    pub turns: u32,
    /// Tokens used by all the model requests of the run, subagents included.
    #[pyo3(get)]
    pub usage: Usage,
    /// Responses API: the id of the last response. With `store=True`, pass it
    /// as `previous_response_id` to continue the conversation.
    #[pyo3(get)]
    pub response_id: Option<String>,
    /// The whole conversation; the run added the messages from `new_messages_start` on.
    pub messages: Vec<Message>,
    pub new_messages_start: usize,
    /// The requests of tool calls still waiting for an answer. When there are
    /// any, the run stopped early: answer them with `execute(answers=...)`.
    #[pyo3(get)]
    pub pending: Vec<InputRequested>,
    /// The state at the end of the run (`execute(state=...)` plus the tools' changes).
    pub state: Value,
}

#[pymethods]
impl TextDelta {
    fn __repr__(&self) -> String {
        match &self.subagent_id {
            Some(subagent_id) => format!("TextDelta(text={:?}, subagent_id={subagent_id:?})", self.text),
            None => format!("TextDelta(text={:?})", self.text),
        }
    }
}

#[pymethods]
impl ToolStarted {
    /// The parsed tool arguments, as a dict.
    #[getter]
    fn arguments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.arguments)
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolStarted(call_id={:?}, name={:?}, arguments={})",
            self.call_id, self.name, self.arguments
        )
    }
}

#[pymethods]
impl ToolFinished {
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolFinished(call_id={:?}, name={:?}, result={:?}, is_error={})",
            self.call_id, self.name, self.result, self.is_error
        )
    }
}

#[pymethods]
impl StateChanged {
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.state)
    }

    #[getter]
    fn delta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.delta)
    }

    fn __repr__(&self) -> String {
        format!("StateChanged(delta={})", self.delta)
    }
}

#[pymethods]
impl SubagentStarted {
    fn __repr__(&self) -> String {
        format!(
            "SubagentStarted(call_id={:?}, name={:?}, task={:?})",
            self.call_id, self.name, self.task
        )
    }
}

#[pymethods]
impl SubagentFinished {
    fn __repr__(&self) -> String {
        format!(
            "SubagentFinished(call_id={:?}, name={:?}, result={:?}, is_error={}, waiting={})",
            self.call_id, self.name, self.result, self.is_error, self.waiting
        )
    }
}

#[pymethods]
impl RunFinished {
    /// The full conversation (without the system prompt), as a list of dicts.
    /// Pass it as `history=` to continue the conversation.
    #[getter]
    fn messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        messages_to_python(py, &self.messages)
    }

    /// Only the messages this run added: its prompt, the answers and the tool
    /// results. Append these to your storage to save the conversation.
    #[getter]
    fn new_messages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        messages_to_python(py, self.added_messages())
    }

    /// The state at the end of the run: save it, and give it back with `execute(state=...)`.
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.state)
    }

    /// `messages` as JSON text, made in Rust: store it without building dicts.
    fn messages_json(&self) -> String {
        serde_json::to_string(&self.messages).expect("messages always serialize")
    }

    /// `new_messages` as JSON text, made in Rust.
    fn new_messages_json(&self) -> String {
        serde_json::to_string(self.added_messages()).expect("messages always serialize")
    }

    fn __repr__(&self) -> String {
        format!(
            "RunFinished(text={:?}, turns={}, usage={}, pending={})",
            self.text,
            self.turns,
            self.usage.__repr__(),
            self.pending.len()
        )
    }
}

impl RunFinished {
    pub fn added_messages(&self) -> &[Message] {
        &self.messages[self.new_messages_start..]
    }
}

fn messages_to_python<'py>(py: Python<'py>, messages: &[Message]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for message in messages {
        let message_json = serde_json::to_value(message).expect("messages always serialize");
        list.append(json_to_python(py, &message_json)?)?;
    }
    Ok(list)
}

/// Any event of a run. The agent loop sends these through a channel; they are
/// turned into Python objects only when Python asks for the next one.
pub enum AgentEvent {
    TextDelta(TextDelta),
    ToolStarted(ToolStarted),
    ToolFinished(ToolFinished),
    InputRequested(InputRequested),
    StateChanged(StateChanged),
    SubagentStarted(SubagentStarted),
    SubagentFinished(SubagentFinished),
    RunFinished(RunFinished),
}

impl AgentEvent {
    pub fn into_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let python_object = match self {
            AgentEvent::TextDelta(event) => Py::new(py, event)?.into_any(),
            AgentEvent::ToolStarted(event) => Py::new(py, event)?.into_any(),
            AgentEvent::ToolFinished(event) => Py::new(py, event)?.into_any(),
            AgentEvent::InputRequested(event) => Py::new(py, event)?.into_any(),
            AgentEvent::StateChanged(event) => Py::new(py, event)?.into_any(),
            AgentEvent::SubagentStarted(event) => Py::new(py, event)?.into_any(),
            AgentEvent::SubagentFinished(event) => Py::new(py, event)?.into_any(),
            AgentEvent::RunFinished(event) => Py::new(py, event)?.into_any(),
        };
        Ok(python_object)
    }

    /// The subagent that sent the event (`None` for the main agent).
    pub fn subagent_id(&self) -> Option<&str> {
        let subagent_id = match self {
            AgentEvent::TextDelta(event) => &event.subagent_id,
            AgentEvent::ToolStarted(event) => &event.subagent_id,
            AgentEvent::ToolFinished(event) => &event.subagent_id,
            AgentEvent::InputRequested(event) => &event.subagent_id,
            AgentEvent::StateChanged(event) => &event.subagent_id,
            AgentEvent::SubagentStarted(event) => &event.subagent_id,
            AgentEvent::SubagentFinished(event) => &event.subagent_id,
            AgentEvent::RunFinished(_) => return None,
        };
        subagent_id.as_deref()
    }
}

/// Where a run sends its events. Python reads them from the other end of the channel.
///
/// A subagent runs with a copy of its caller's sender made by `for_subagent`:
/// its events go to the same channel, marked with the subagent's id.
#[derive(Clone)]
pub struct EventSender {
    channel: Sender<PyResult<AgentEvent>>,
    /// `None` for the main agent.
    subagent_id: Option<String>,
}

impl EventSender {
    pub fn new(channel: Sender<PyResult<AgentEvent>>) -> Self {
        EventSender {
            channel,
            subagent_id: None,
        }
    }

    /// The sender for the subagent that the tool call `call_id` runs.
    pub fn for_subagent(&self, call_id: &str) -> Self {
        EventSender {
            channel: self.channel.clone(),
            subagent_id: Some(self.app_call_id(call_id)),
        }
    }

    /// The id the app sees for a tool call of this agent: `call_1` for the
    /// main agent, `call_0/call_1` inside the subagent `call_0`.
    pub fn app_call_id(&self, call_id: &str) -> String {
        match &self.subagent_id {
            Some(subagent_id) => format!("{subagent_id}/{call_id}"),
            None => call_id.to_string(),
        }
    }

    /// Marks `event` as this agent's: its subagent id, and the app's call ids.
    fn label(&self, event: &mut AgentEvent) {
        if self.subagent_id.is_none() {
            return; // the main agent's events are already right
        }
        let subagent_id = self.subagent_id.clone();
        match event {
            AgentEvent::TextDelta(event) => event.subagent_id = subagent_id,
            AgentEvent::ToolStarted(event) => {
                event.call_id = self.app_call_id(&event.call_id);
                event.subagent_id = subagent_id;
            }
            AgentEvent::ToolFinished(event) => {
                event.call_id = self.app_call_id(&event.call_id);
                event.subagent_id = subagent_id;
            }
            AgentEvent::InputRequested(event) => {
                event.call_id = self.app_call_id(&event.call_id);
                event.subagent_id = subagent_id;
            }
            AgentEvent::StateChanged(event) => event.subagent_id = subagent_id,
            AgentEvent::SubagentStarted(event) => {
                event.call_id = self.app_call_id(&event.call_id);
                event.subagent_id = subagent_id;
            }
            AgentEvent::SubagentFinished(event) => {
                event.call_id = self.app_call_id(&event.call_id);
                event.subagent_id = subagent_id;
            }
            AgentEvent::RunFinished(_) => {}
        }
    }

    pub async fn send(&self, mut event: AgentEvent) {
        self.label(&mut event);
        // If Python stopped listening, sending fails; the run is being cancelled anyway.
        let _ = self.channel.send(Ok(event)).await;
    }

    /// Sends a new request, and returns it as the app sees it (for `on_request`).
    pub async fn send_request(&self, request: InputRequested) -> InputRequested {
        let mut event = AgentEvent::InputRequested(request);
        self.label(&mut event);
        let AgentEvent::InputRequested(request) = event else {
            unreachable!("labelling keeps the kind of event");
        };
        let _ = self.channel.send(Ok(AgentEvent::InputRequested(request.clone()))).await;
        request
    }

    /// Ends the stream with an exception for Python.
    pub async fn send_error(&self, error: PyErr) {
        let _ = self.channel.send(Err(error)).await;
    }
}
