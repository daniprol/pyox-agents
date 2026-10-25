//! The events Python receives while iterating over a run.
//!
//! Each event is a Rust struct exposed as a read-only Python class:
//! - `#[pyclass(frozen)]` makes the class immutable from Python.
//! - `get_all` creates a Python property for every field.
//!
//! Fields stay in Rust until Python reads them, and there is no per-object
//! `__dict__`, so events are small and cheap to create.

use pyo3::prelude::*;
use pyo3::types::PyList;
use serde_json::Value;

use crate::json_convert::json_to_python;
use crate::messages::Message;

/// A piece of the assistant's answer, streamed as soon as it arrives.
#[pyclass(frozen, get_all, module = "pyox_agents")]
pub struct TextDelta {
    pub text: String,
}

/// The model asked to call a tool and the tool is about to run.
#[pyclass(frozen, module = "pyox_agents")]
pub struct ToolStarted {
    #[pyo3(get)]
    pub call_id: String,
    #[pyo3(get)]
    pub name: String,
    pub arguments: Value,
}

/// A tool finished. `is_error` is true if it raised an exception.
#[pyclass(frozen, get_all, module = "pyox_agents")]
pub struct ToolFinished {
    pub call_id: String,
    pub name: String,
    pub result: String,
    pub is_error: bool,
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
    #[pyo3(get)]
    pub input_tokens: u64,
    #[pyo3(get)]
    pub output_tokens: u64,
    /// The whole conversation; the run added the messages from `new_messages_start` on.
    pub messages: Vec<Message>,
    pub new_messages_start: usize,
}

#[pymethods]
impl TextDelta {
    fn __repr__(&self) -> String {
        format!("TextDelta(text={:?})", self.text)
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
    fn __repr__(&self) -> String {
        format!(
            "ToolFinished(call_id={:?}, name={:?}, result={:?}, is_error={})",
            self.call_id, self.name, self.result, self.is_error
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
            "RunFinished(text={:?}, turns={}, input_tokens={}, output_tokens={})",
            self.text, self.turns, self.input_tokens, self.output_tokens
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
    RunFinished(RunFinished),
}

impl AgentEvent {
    pub fn into_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let python_object = match self {
            AgentEvent::TextDelta(event) => Py::new(py, event)?.into_any(),
            AgentEvent::ToolStarted(event) => Py::new(py, event)?.into_any(),
            AgentEvent::ToolFinished(event) => Py::new(py, event)?.into_any(),
            AgentEvent::RunFinished(event) => Py::new(py, event)?.into_any(),
        };
        Ok(python_object)
    }
}
