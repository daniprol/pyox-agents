//! Tools: what the model can call.
//!
//! A tool is implemented either in Python or in Rust:
//!
//! - **Python tools** (`pyox_agents.Tool`, usually made with `@tool`). Any object
//!   with `name`, `description`, `parameters` and an async `invoke(arguments)`
//!   method works. That includes Rust functions compiled in a separate PyO3
//!   package and wrapped with `Tool(...)`. A tool with `takes_ctx = True` is
//!   called as `invoke(arguments, ctx)`, to give it the run's `RunContext`.
//! - **Native Rust tools** from a separately compiled package made with the
//!   `pyox-tools` crate. They carry pointers to their Rust functions, which
//!   the engine calls directly, without Python (see `pyox_tools::native`).
//! - **Frontend tools** (`@tool(frontend=True)`, `Tool.frontend_tool(...)`). They have
//!   no code here: the app runs them and sends back the result (see `human_input`).
//! - **Subagents** (`agent.as_tool(...)`): another agent, run by the agent loop (see `subagents`).
//!
//! Any tool object may have a `requires_approval` attribute: `True`, or a
//! function `(ctx, arguments) -> bool` deciding for each call, and an
//! `on_error` attribute: a function `(ctx, call, error)` choosing what the
//! model reads when the tool fails (see `tool_call::failure_result`).
//!
//! A failing tool never stops the run by itself: the model gets an error
//! result and decides what to do. `ToolDefinition::run` returns the failure
//! as a Python exception (`Err(PyErr)`), so the `on_error` and `on_tool_error`
//! functions see the real exception.

use std::time::Duration;

use pyo3::exceptions::{PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyox_tools::native::NativeTool;
use serde_json::{Map, Value};

/// The result of running a tool, as sent back to the model.
pub use pyox_tools::hooks::ToolResult;

use crate::errors::{InvalidArguments, ToolFailed};
use crate::human_input::{SharedCallLog, new_request};
use crate::json_convert::{json_to_python, python_to_json};
use crate::messages::{Content, Request, RequestKind, ToolCall, content_from_python};
use crate::python_call::{PythonCalls, call_python};
use crate::run_context::Context;
use crate::subagents::{Subagent, subagent_tool_from_python};

/// Everything the agent loop needs to know about one tool.
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON schema describing the arguments.
    pub parameters: Value,
    pub implementation: ToolImplementation,
    pub approval: Approval,
    /// The tool's own `on_error(ctx, call, error)` function, if it has one.
    pub on_error: Option<Py<PyAny>>,
    /// How long a call may take (`timeout_seconds`), if limited.
    pub timeout: Option<Duration>,
}

/// Whether a call must be approved by a person before the tool runs.
pub enum Approval {
    Never,
    Always,
    /// A Python function `(ctx, arguments) -> bool` (sync or async) decides for each call.
    WhenFunctionSays(Py<PyAny>),
}

/// What runs when the model calls a tool.
pub enum ToolImplementation {
    /// A Python or Rust function (`ToolDefinition::run_code`).
    Code(ToolCode),
    /// No code here: the app runs the tool and replies with the result.
    Frontend,
    /// Another agent (see `subagents`).
    Subagent(Subagent),
    /// A subagent's `ask_user(question)`: the person's answer is the result.
    AskUser,
    /// A subagent's `ask_parent(question)`: its run stops, and the calling agent gets the question.
    AskParent,
}

/// A tool function pyox calls itself.
pub enum ToolCode {
    /// A Python object with an `invoke(arguments: dict) -> str` method (sync or async).
    /// `takes_context`: call it as `invoke(arguments, ctx)` instead.
    /// `run_in_thread`: a sync run calls it on a worker thread, not the thread reading the run.
    Python {
        tool: Py<PyAny>,
        takes_context: bool,
        run_in_thread: bool,
    },
    /// A tool written in Rust in a separately compiled package.
    Native(NativeTool),
}

impl ToolImplementation {
    /// The kind's name, for the tool call's span (`pyox.tool.kind`).
    pub fn kind(&self) -> &'static str {
        match self {
            ToolImplementation::Code(ToolCode::Python { .. }) => "python",
            ToolImplementation::Code(ToolCode::Native(_)) => "rust",
            ToolImplementation::Frontend => "frontend",
            ToolImplementation::Subagent(_) => "subagent",
            ToolImplementation::AskUser => "ask_user",
            ToolImplementation::AskParent => "ask_parent",
        }
    }
}

/// Whether the tool has the attribute `name`, set to something true (`tool.name` in Python).
fn is_set(tool: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    match tool.getattr(name) {
        Ok(value) => value.is_truthy(),
        Err(_) => Ok(false),
    }
}

/// A tool's `timeout_seconds`, checked.
fn timeout_from_seconds(seconds: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .ok()
        .filter(|timeout| !timeout.is_zero())
        .ok_or_else(|| PyValueError::new_err(format!("`timeout_seconds` must be a positive number, got {seconds}")))
}

/// `pyox_agents.ToolFailed(message)`: a tool failure whose message the model reads as it is.
pub fn tool_failed(message: impl Into<String>) -> PyErr {
    ToolFailed::new_err(message.into())
}

/// `pyox_agents.InvalidArguments(tool_name, [problem])`: the model's arguments do not fit the tool.
pub fn invalid_arguments(tool_name: &str, problem: String) -> PyErr {
    InvalidArguments::new_err((tool_name.to_string(), vec![problem]))
}

/// A tool result telling the model something went wrong.
pub fn error_result(message: impl Into<String>) -> ToolResult {
    ToolResult {
        content: Content::Text(message.into()),
        is_error: true,
        metadata: Value::Null,
    }
}

/// `content` is text (a `String`), or a `Content` with images or files.
pub fn success_result(content: impl Into<Content>, metadata: Value) -> ToolResult {
    ToolResult {
        content: content.into(),
        is_error: false,
        metadata,
    }
}

impl ToolDefinition {
    /// Reads a tool given from Python: a `Subagent`, a Rust tool or a Python tool object.
    pub fn from_python(tool: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(subagent_tool) = subagent_tool_from_python(tool)? {
            return Ok(subagent_tool);
        }
        let approval = match tool.getattr("requires_approval") {
            Ok(requires_approval) if requires_approval.is_callable() => {
                Approval::WhenFunctionSays(requires_approval.unbind())
            }
            Ok(requires_approval) if requires_approval.is_truthy()? => Approval::Always,
            _ => Approval::Never,
        };

        let implementation = match NativeTool::from_python(tool)? {
            Some(native_tool) => ToolImplementation::Code(ToolCode::Native(native_tool)),
            None if is_set(tool, "frontend")? => ToolImplementation::Frontend,
            None if tool.hasattr("invoke")? => ToolImplementation::Code(ToolCode::Python {
                tool: tool.clone().unbind(),
                takes_context: is_set(tool, "takes_ctx")?,
                run_in_thread: is_set(tool, "run_in_thread")?,
            }),
            None => {
                return Err(PyValueError::new_err(format!(
                    "{} is not a tool. Use the @tool decorator or Tool(...).",
                    tool.repr()?
                )));
            }
        };
        // `getattr` with a default: tools without `on_error` (or with `None`) have none.
        let on_error = tool.getattr("on_error").ok().filter(|function| !function.is_none());
        let timeout = match tool.getattr("timeout_seconds") {
            Ok(seconds) if !seconds.is_none() => Some(timeout_from_seconds(seconds.extract()?)?),
            _ => None,
        };
        Ok(ToolDefinition {
            name: tool.getattr("name")?.extract()?,
            description: tool.getattr("description")?.extract()?,
            parameters: python_to_json(&tool.getattr("parameters")?)?,
            implementation,
            approval,
            on_error: on_error.map(Bound::unbind),
            timeout,
        })
    }

    /// A tool the app runs, sent for one run (`stream(frontend_tools=[...])`):
    /// a dict with `name`, and optionally `description` and `parameters`.
    pub fn frontend_from_python(tool: &Bound<'_, PyAny>) -> PyResult<Self> {
        let field = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            let value = tool.get_item(key).ok().filter(|value| !value.is_none());
            Ok(value)
        };
        let Some(name) = field("name")? else {
            return Err(PyValueError::new_err(format!(
                "A frontend tool needs a \"name\": {}",
                tool.repr()?
            )));
        };
        Ok(ToolDefinition {
            name: name.extract()?,
            description: field("description")?
                .map(|text| text.extract())
                .transpose()?
                .unwrap_or_default(),
            parameters: match field("parameters")? {
                Some(parameters) => python_to_json(&parameters)?,
                None => serde_json::json!({"type": "object", "properties": {}}),
            },
            implementation: ToolImplementation::Frontend,
            approval: Approval::Never,
            on_error: None,
            timeout: None,
        })
    }

    /// The approval request this call needs, if any. `Err` is the exception
    /// the `requires_approval` function raised.
    pub async fn approval_request(
        &self,
        call_log: &SharedCallLog,
        context: &Context,
        python_calls: &PythonCalls,
    ) -> PyResult<Option<Request>> {
        let needs_approval = match &self.approval {
            Approval::Never => false,
            Approval::Always => true,
            Approval::WhenFunctionSays(function) => {
                let function = Python::attach(|py| function.clone_ref(py));
                let (context, call_log, ctx_calls) = (context.clone(), call_log.clone(), python_calls.clone());
                let arguments = call_log.lock().unwrap().call.arguments.clone();
                let answer = call_python(
                    function,
                    move |py| {
                        let ctx = context
                            .to_python(py, Some(call_log), ctx_calls)?
                            .into_bound(py)
                            .into_any();
                        Ok(vec![ctx, json_to_python(py, &arguments)?])
                    },
                    python_calls,
                )
                .await?;
                Python::attach(|py| answer.bind(py).is_truthy())?
            }
        };
        Ok(needs_approval.then(|| new_request(RequestKind::Approval, Value::Null, Value::Null)))
    }

    /// Runs the tool's `code` (a Python or Rust function), giving up after
    /// its `timeout` with a `TimeoutError`; giving up drops the call, so an
    /// async Python tool is cancelled.
    pub async fn run_code(
        &self,
        code: &ToolCode,
        call: &ToolCall,
        call_log: &SharedCallLog,
        context: &Context,
        python_calls: &PythonCalls,
    ) -> PyResult<ToolResult> {
        let running = code.run(&self.name, call, call_log, context, python_calls);
        let Some(timeout) = self.timeout else {
            return running.await;
        };
        match tokio::time::timeout(timeout, running).await {
            Ok(outcome) => outcome,
            Err(_) => Err(PyTimeoutError::new_err(format!(
                "Tool {:?} timed out after {} seconds",
                self.name,
                timeout.as_secs_f64()
            ))),
        }
    }
}

impl ToolCode {
    /// Calls the function. `Err` is the exception it failed with: a Python
    /// tool's own exception, or a `ToolFailed` for a Rust tool's error message.
    /// `call_log` is what the call asked so far (Python tools see it through `ctx`).
    async fn run(
        &self,
        tool_name: &str,
        call: &ToolCall,
        call_log: &SharedCallLog,
        context: &Context,
        python_calls: &PythonCalls,
    ) -> PyResult<ToolResult> {
        let Value::Object(arguments) = &call.arguments else {
            // Arguments that are not valid JSON are kept as the model's text.
            let problem = format!("the arguments must be a JSON object, got {}", call.arguments);
            return Err(invalid_arguments(tool_name, problem));
        };

        match self {
            ToolCode::Native(native_tool) => {
                let native_tool = *native_tool;
                let arguments = Value::Object(arguments.clone());
                run_rust_tool(tool_name, move || native_tool.call(&arguments)).await
            }
            ToolCode::Python {
                tool,
                takes_context,
                run_in_thread,
            } => {
                let context = takes_context.then(|| (context.clone(), call_log.clone()));
                let python_calls = match run_in_thread {
                    true => python_calls.on_worker_thread(),
                    false => python_calls.clone(),
                };
                run_python_tool(tool, arguments, context, &python_calls).await
            }
        }
    }
}

/// Runs a Rust tool on a thread meant for blocking work, so slow tools
/// (file access, heavy computation) do not block the network tasks.
/// No Python is involved.
async fn run_rust_tool(
    tool_name: &str,
    run: impl FnOnce() -> Result<String, String> + Send + 'static,
) -> PyResult<ToolResult> {
    match tokio::task::spawn_blocking(run).await {
        Ok(Ok(content)) => Ok(success_result(content, Value::Null)),
        Ok(Err(message)) => Err(tool_failed(message)),
        Err(_) => Err(tool_failed(format!("Tool {tool_name} crashed"))),
    }
}

/// Calls `tool.invoke(arguments)`, or `tool.invoke(arguments, ctx)` when a
/// context is given, awaiting the result if the tool is async.
/// `invoke` returns text, or a `ToolReturn`: content (text, or a list of text,
/// images and files) plus metadata for the app.
async fn run_python_tool(
    python_tool: &Py<PyAny>,
    arguments: &Map<String, Value>,
    context: Option<(Context, SharedCallLog)>,
    python_calls: &PythonCalls,
) -> PyResult<ToolResult> {
    let invoke = Python::attach(|py| python_tool.bind(py).getattr("invoke").map(Bound::unbind))?;
    let arguments = Value::Object(arguments.clone());
    let ctx_calls = python_calls.clone();
    let tool_result = call_python(
        invoke,
        move |py| {
            let mut python_arguments = vec![json_to_python(py, &arguments)?];
            if let Some((context, call_log)) = context {
                let ctx = context.to_python(py, Some(call_log), ctx_calls)?;
                python_arguments.push(ctx.into_bound(py).into_any());
            }
            Ok(python_arguments)
        },
        python_calls,
    )
    .await?;
    Python::attach(|py| {
        let tool_result = tool_result.bind(py);
        // Text, or a list of text, images and files; otherwise a `ToolReturn`.
        if !tool_result.hasattr("metadata")? {
            return Ok(success_result(content_from_python(tool_result)?, Value::Null));
        }
        Ok(success_result(
            content_from_python(&tool_result.getattr("content")?)?,
            python_to_json(&tool_result.getattr("metadata")?)?,
        ))
    })
}
