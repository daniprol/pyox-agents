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
//! function `(ctx, arguments) -> bool` deciding for each call.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use pyox_tools::native::NativeTool;
use serde_json::{Map, Value};

/// The result of running a tool, as sent back to the model.
pub use pyox_tools::hooks::ToolResult;

use crate::human_input::{SharedCallLog, new_request};
use crate::json_convert::{json_to_python, python_to_json};
use crate::messages::{Content, Request, RequestKind, ToolCall, content_from_python};
use crate::python_call::call_python;
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
}

/// Whether a call must be approved by a person before the tool runs.
pub enum Approval {
    Never,
    Always,
    /// A Python function `(ctx, arguments) -> bool` (sync or async) decides for each call.
    WhenFunctionSays(Py<PyAny>),
}

/// Where the code of a tool lives.
pub enum ToolImplementation {
    /// A Python object with an `invoke(arguments: dict) -> str` method (sync or async).
    /// `takes_context`: call it as `invoke(arguments, ctx)` instead.
    Python { tool: Py<PyAny>, takes_context: bool },
    /// A tool written in Rust in a separately compiled package.
    Native(NativeTool),
    /// No code here: the app runs the tool and replies with the result.
    Frontend,
    /// Another agent (see `subagents`).
    Subagent(Subagent),
    /// A subagent's `ask_user(question)`: the person's answer is the result.
    AskUser,
    /// A subagent's `ask_parent(question)`: its run stops, and the calling agent gets the question.
    AskParent,
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

        let is_frontend_tool = match tool.getattr("frontend") {
            Ok(frontend) => frontend.is_truthy()?,
            Err(_) => false,
        };
        let implementation = match NativeTool::from_python(tool)? {
            Some(native_tool) => ToolImplementation::Native(native_tool),
            None if is_frontend_tool => ToolImplementation::Frontend,
            None if tool.hasattr("invoke")? => ToolImplementation::Python {
                tool: tool.clone().unbind(),
                takes_context: match tool.getattr("takes_ctx") {
                    Ok(takes_ctx) => takes_ctx.is_truthy()?,
                    Err(_) => false,
                },
            },
            None => {
                return Err(PyValueError::new_err(format!(
                    "{} is not a tool. Use the @tool decorator or Tool(...).",
                    tool.repr()?
                )));
            }
        };
        Ok(ToolDefinition {
            name: tool.getattr("name")?.extract()?,
            description: tool.getattr("description")?.extract()?,
            parameters: python_to_json(&tool.getattr("parameters")?)?,
            implementation,
            approval,
        })
    }

    /// A tool the app runs, sent for one run (`execute(frontend_tools=[...])`):
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
        })
    }

    /// The approval request this call needs, if any. `Err` is the error
    /// result for the model when the `requires_approval` function failed.
    pub async fn approval_request(
        &self,
        call_log: &SharedCallLog,
        context: &Context,
        python_task_locals: &TaskLocals,
    ) -> Result<Option<Request>, ToolResult> {
        let needs_approval = match &self.approval {
            Approval::Never => false,
            Approval::Always => true,
            Approval::WhenFunctionSays(function) => {
                let function = Python::attach(|py| function.clone_ref(py));
                let (context, call_log) = (context.clone(), call_log.clone());
                let arguments = call_log.lock().unwrap().call.arguments.clone();
                let answer = call_python(
                    function,
                    move |py| {
                        let ctx = context.to_python(py, Some(call_log))?.into_bound(py).into_any();
                        Ok(vec![ctx, json_to_python(py, &arguments)?])
                    },
                    python_task_locals,
                )
                .await
                .and_then(|answer| Python::attach(|py| answer.bind(py).is_truthy()));
                answer.map_err(|python_error| error_result(python_error.to_string()))?
            }
        };
        Ok(needs_approval.then(|| new_request(RequestKind::Approval, Value::Null, Value::Null)))
    }

    /// Runs the tool. Errors never stop the agent: they are returned to the
    /// model as an error result, so it can react (for example, retry).
    /// `call_log` is what the call asked so far (Python tools see it through `ctx`).
    pub async fn run(
        &self,
        call: &ToolCall,
        call_log: &SharedCallLog,
        context: &Context,
        python_task_locals: &TaskLocals,
    ) -> ToolResult {
        let Value::Object(arguments) = &call.arguments else {
            return error_result(format!("Tool arguments must be a JSON object, got: {}", call.arguments));
        };

        match &self.implementation {
            ToolImplementation::Native(native_tool) => {
                let native_tool = *native_tool;
                let arguments = Value::Object(arguments.clone());
                run_rust_tool(&self.name, move || native_tool.call(&arguments)).await
            }
            ToolImplementation::Python { tool, takes_context } => {
                let context = takes_context.then(|| (context.clone(), call_log.clone()));
                match run_python_tool(tool, arguments, context, python_task_locals).await {
                    Ok(result) => result,
                    // `to_string()` gives e.g. "ValueError: city not found".
                    Err(python_error) => error_result(python_error.to_string()),
                }
            }
            ToolImplementation::Frontend
            | ToolImplementation::Subagent(_)
            | ToolImplementation::AskUser
            | ToolImplementation::AskParent => unreachable!("the agent loop runs these itself"),
        }
    }
}

/// Runs a Rust tool on a thread meant for blocking work, so slow tools
/// (file access, heavy computation) do not block the network tasks.
/// No Python is involved.
async fn run_rust_tool(tool_name: &str, run: impl FnOnce() -> Result<String, String> + Send + 'static) -> ToolResult {
    match tokio::task::spawn_blocking(run).await {
        Ok(Ok(content)) => success_result(content, Value::Null),
        Ok(Err(message)) => error_result(message),
        Err(_) => error_result(format!("Tool {tool_name} crashed")),
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
    python_task_locals: &TaskLocals,
) -> PyResult<ToolResult> {
    let invoke = Python::attach(|py| python_tool.bind(py).getattr("invoke").map(Bound::unbind))?;
    let arguments = Value::Object(arguments.clone());
    let tool_result = call_python(
        invoke,
        move |py| {
            let mut python_arguments = vec![json_to_python(py, &arguments)?];
            if let Some((context, call_log)) = context {
                python_arguments.push(context.to_python(py, Some(call_log))?.into_bound(py).into_any());
            }
            Ok(python_arguments)
        },
        python_task_locals,
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
