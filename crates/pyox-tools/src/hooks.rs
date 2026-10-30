//! Agent hooks in Rust: functions the agent calls at fixed points of a run,
//! directly in Rust, without going through Python.
//!
//! Register a closure for each point you care about:
//!
//! ```ignore
//! use pyox_tools::hooks::{Hooks, add_hooks};
//!
//! #[pymodule]
//! fn my_hooks(module: &Bound<'_, PyModule>) -> PyResult<()> {
//!     let no_shell = Hooks::new().before_tool_call(|_context, call| {
//!         if call.name == "shell" {
//!             return Err("Shell commands are disabled".into()); // the model sees this
//!         }
//!         Ok(())
//!     });
//!     add_hooks(module, "no_shell", no_shell)
//! }
//! ```
//!
//! In Python: `Agent(hooks=[my_hooks.no_shell])`.
//!
//! The hooks, in the order a run reaches them:
//!
//! | Hook                   | Receives                    | Can change             |
//! |------------------------|-----------------------------|------------------------|
//! | `before_run`           | -                           | -                      |
//! | `before_model_request` | the messages to send        | the messages to send   |
//! | `after_model_response` | the assistant message       | the assistant message  |
//! | `before_tool_call`     | the tool call               | the tool call          |
//! | `after_tool_call`      | the tool call and result    | the result             |
//! | `after_run`            | the whole conversation      | -                      |
//!
//! Changes made in `before_model_request` only affect that one request: the
//! history (and what history stores save) stays as it was.
//!
//! Returning `Err(message)` from `before_tool_call` or `after_tool_call`
//! blocks the tool: the model gets the message as the tool's error result.
//! From any other hook, it stops the run with an `AgentError`.

use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::messages::{Message, ToolCall};
use crate::native_hooks::{self, Hook};
use crate::usage::Usage;

/// What a hook knows about the run it is part of.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunContext {
    /// The conversation given with `execute(conversation_id=...)`, if any.
    pub conversation_id: Option<String>,
    /// The current turn: 1 during the first model request, 2 during the second...
    pub turn: u32,
    /// Tokens used by the run so far.
    pub usage: Usage,
}

/// The result of a tool, as sent back to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
    /// Data for the app, never sent to the model (`ToolReturn(metadata=...)`).
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

/// What every hook returns: `Err(message)` to block a tool or stop the run.
pub type HookResult = Result<(), String>;

// One type per hook signature. `Box<dyn Fn ...>` holds any closure;
// `Send + Sync` lets runs on different threads call it at the same time.
type BeforeRun = Box<dyn Fn(&RunContext) -> HookResult + Send + Sync>;
type BeforeModelRequest = Box<dyn Fn(&RunContext, &mut Vec<Message>) -> HookResult + Send + Sync>;
type AfterModelResponse = Box<dyn Fn(&RunContext, &mut Message) -> HookResult + Send + Sync>;
type BeforeToolCall = Box<dyn Fn(&RunContext, &mut ToolCall) -> HookResult + Send + Sync>;
type AfterToolCall = Box<dyn Fn(&RunContext, &ToolCall, &mut ToolResult) -> HookResult + Send + Sync>;
type AfterRun = Box<dyn Fn(&RunContext, &[Message]) -> HookResult + Send + Sync>;

/// A set of hooks. Each method registers one; leave out the ones you don't need.
#[derive(Default)]
pub struct Hooks {
    before_run: Option<BeforeRun>,
    before_model_request: Option<BeforeModelRequest>,
    after_model_response: Option<AfterModelResponse>,
    before_tool_call: Option<BeforeToolCall>,
    after_tool_call: Option<AfterToolCall>,
    after_run: Option<AfterRun>,
}

impl Hooks {
    pub fn new() -> Self {
        Hooks::default()
    }

    /// Called once when a run starts.
    pub fn before_run(mut self, hook: impl Fn(&RunContext) -> HookResult + Send + Sync + 'static) -> Self {
        self.before_run = Some(Box::new(hook));
        self
    }

    /// Called before each model request, with the messages about to be sent.
    pub fn before_model_request(
        mut self,
        hook: impl Fn(&RunContext, &mut Vec<Message>) -> HookResult + Send + Sync + 'static,
    ) -> Self {
        self.before_model_request = Some(Box::new(hook));
        self
    }

    /// Called after each model response, with the assistant message
    /// (its text was already streamed; its tool calls haven't run yet).
    pub fn after_model_response(
        mut self,
        hook: impl Fn(&RunContext, &mut Message) -> HookResult + Send + Sync + 'static,
    ) -> Self {
        self.after_model_response = Some(Box::new(hook));
        self
    }

    /// Called before each tool runs. Change `call.arguments` to change what the
    /// tool receives, or return `Err` to block it.
    pub fn before_tool_call(
        mut self,
        hook: impl Fn(&RunContext, &mut ToolCall) -> HookResult + Send + Sync + 'static,
    ) -> Self {
        self.before_tool_call = Some(Box::new(hook));
        self
    }

    /// Called after each tool runs, with the result the model is about to get.
    pub fn after_tool_call(
        mut self,
        hook: impl Fn(&RunContext, &ToolCall, &mut ToolResult) -> HookResult + Send + Sync + 'static,
    ) -> Self {
        self.after_tool_call = Some(Box::new(hook));
        self
    }

    /// Called once when a run finishes successfully, with the whole conversation.
    pub fn after_run(mut self, hook: impl Fn(&RunContext, &[Message]) -> HookResult + Send + Sync + 'static) -> Self {
        self.after_run = Some(Box::new(hook));
        self
    }

    /// Whether a hook was registered, so the agent can skip the others.
    pub(crate) fn has(&self, hook: Hook) -> bool {
        match hook {
            Hook::BeforeRun => self.before_run.is_some(),
            Hook::BeforeModelRequest => self.before_model_request.is_some(),
            Hook::AfterModelResponse => self.after_model_response.is_some(),
            Hook::BeforeToolCall => self.before_tool_call.is_some(),
            Hook::AfterToolCall => self.after_tool_call.is_some(),
            Hook::AfterRun => self.after_run.is_some(),
        }
    }

    /// Calls one hook with its arguments as JSON values (the same arguments a
    /// Python hook gets after `ctx`). Returns the JSON of what the hook may
    /// change (its last argument), or empty text for hooks that change nothing.
    pub(crate) fn call(&self, hook: Hook, context: &RunContext, arguments: Vec<Value>) -> Result<String, String> {
        let mut arguments = arguments.into_iter();
        match hook {
            Hook::BeforeRun => {
                if let Some(before_run) = &self.before_run {
                    before_run(context)?;
                }
                Ok(String::new())
            }
            Hook::BeforeModelRequest => {
                let mut messages: Vec<Message> = next_argument(&mut arguments)?;
                if let Some(before_model_request) = &self.before_model_request {
                    before_model_request(context, &mut messages)?;
                }
                Ok(to_json(&messages))
            }
            Hook::AfterModelResponse => {
                let mut message: Message = next_argument(&mut arguments)?;
                if let Some(after_model_response) = &self.after_model_response {
                    after_model_response(context, &mut message)?;
                }
                Ok(to_json(&message))
            }
            Hook::BeforeToolCall => {
                let mut call: ToolCall = next_argument(&mut arguments)?;
                if let Some(before_tool_call) = &self.before_tool_call {
                    before_tool_call(context, &mut call)?;
                }
                Ok(to_json(&call))
            }
            Hook::AfterToolCall => {
                let call: ToolCall = next_argument(&mut arguments)?;
                let mut result: ToolResult = next_argument(&mut arguments)?;
                if let Some(after_tool_call) = &self.after_tool_call {
                    after_tool_call(context, &call, &mut result)?;
                }
                Ok(to_json(&result))
            }
            Hook::AfterRun => {
                let messages: Vec<Message> = next_argument(&mut arguments)?;
                if let Some(after_run) = &self.after_run {
                    after_run(context, &messages)?;
                }
                Ok(String::new())
            }
        }
    }
}

fn next_argument<Argument: DeserializeOwned>(arguments: &mut impl Iterator<Item = Value>) -> Result<Argument, String> {
    let argument = arguments.next().ok_or("A hook argument is missing")?;
    serde_json::from_value(argument).map_err(|error| format!("Invalid hook argument: {error}"))
}

fn to_json(value: &impl Serialize) -> String {
    serde_json::to_string(value).expect("hook arguments always serialize")
}

/// The Python object for a set of Rust hooks: pass it in `Agent(hooks=[...])`.
/// pyox-agents finds the Rust functions through `__pyox_hooks__` and calls
/// them without Python.
#[pyclass(frozen, name = "RustHooks")]
pub struct RustHooks {
    /// Owns the hooks: they are freed when the last reference to it goes away.
    native_hooks: Py<PyCapsule>,
}

impl RustHooks {
    /// Wraps your hooks so they can be returned to Python.
    pub fn new(hooks: Hooks) -> PyResult<Self> {
        Python::attach(|py| {
            Ok(RustHooks {
                native_hooks: native_hooks::capsule_for(py, hooks)?.unbind(),
            })
        })
    }
}

#[pymethods]
impl RustHooks {
    /// The capsule with the hooks' native functions (see `native_hooks`).
    #[getter]
    fn __pyox_hooks__(&self, py: Python<'_>) -> Py<PyCapsule> {
        self.native_hooks.clone_ref(py)
    }

    fn __repr__(&self) -> &'static str {
        "RustHooks()"
    }
}

/// Adds a set of hooks to a PyO3 module as an attribute called `name`.
pub fn add_hooks(module: &Bound<'_, PyModule>, name: &str, hooks: Hooks) -> PyResult<()> {
    module.add(name, Py::new(module.py(), RustHooks::new(hooks)?)?)
}
