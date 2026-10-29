//! Our own conversation history format.
//!
//! `Message` and `ToolCall` are defined in the shared `pyox-tools` crate, so
//! Rust history stores compiled separately use the very same types. See
//! `pyox_tools::messages` for the format and how it looks in Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
pub use pyox_tools::messages::{ContentPart, Message, Reasoning, ToolCall, UserContent};
use serde_json::Value;

use crate::json_convert::python_to_json;

/// Reads a prompt given from Python: a `str`, or a list of strings and part
/// dicts (as made by `pyox_agents.image(...)` and `file(...)`).
pub fn prompt_from_python(prompt: &Bound<'_, PyAny>) -> PyResult<UserContent> {
    if let Ok(text) = prompt.extract::<String>() {
        return Ok(UserContent::Text(text));
    }
    let Value::Array(items) = python_to_json(prompt)? else {
        return Err(PyValueError::new_err("The prompt must be a str or a list"));
    };
    let parts = items
        .into_iter()
        .map(|item| match item {
            // A plain string in the list is a text part.
            Value::String(text) => Ok(ContentPart::Text { text }),
            part => serde_json::from_value(part)
                .map_err(|error| PyValueError::new_err(format!("Invalid prompt part: {error}"))),
        })
        .collect::<PyResult<_>>()?;
    Ok(UserContent::Parts(parts))
}

/// The APIs want tool arguments as JSON text. Invalid arguments were stored
/// as a plain string (see `ToolCall`), so we send that string back unchanged.
pub fn arguments_as_text(arguments: &Value) -> String {
    match arguments {
        Value::String(raw_text) => raw_text.clone(),
        other => other.to_string(),
    }
}

/// The opposite of `arguments_as_text`: the arguments text the model sent,
/// parsed. If it is not valid JSON we keep the text, so the model can see
/// (and fix) its mistake.
pub fn arguments_from_text(arguments_text: String) -> Value {
    if arguments_text.trim().is_empty() {
        return Value::Object(Default::default()); // a tool without arguments: `{}`
    }
    serde_json::from_str(&arguments_text).unwrap_or(Value::String(arguments_text))
}

/// Token counts reported by the API.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// The part of `input_tokens` read from OpenAI's prompt cache (cheaper and faster).
    pub cached_input_tokens: u64,
}

impl TokenUsage {
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
    }
}
