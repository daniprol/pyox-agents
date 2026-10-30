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
//! - **Built-in Rust tools** (`BuiltinTool("read_file")`). They are compiled into
//!   this library and run entirely in Rust, without touching Python.
//! - **Frontend tools** (`@tool(frontend=True)`, `Tool.frontend_tool(...)`). They have
//!   no code here: the app runs them and sends back the result (see `human_input`).
//!
//! Any tool object may have a `requires_approval` attribute: `True`, or a
//! function `(ctx, arguments) -> bool` deciding for each call.

mod builtin;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use pyox_tools::native::NativeTool;
use serde_json::{Map, Value};

pub use builtin::BuiltinTool;
/// The result of running a tool, as sent back to the model.
pub use pyox_tools::hooks::ToolResult;

use crate::human_input::{SharedCallLog, new_request};
use crate::json_convert::{json_to_python, python_to_json};
use crate::messages::{Request, RequestKind, ToolCall};
use crate::python_call::call_python;
use crate::run_context::Context;

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
    /// A tool written in Rust inside this library.
    Builtin(&'static pyox_tools::ToolDefinition),
    /// A tool written in Rust in a separately compiled package.
    Native(NativeTool),
    /// No code here: the app runs the tool and replies with the result.
    Frontend,
}

/// A tool result telling the model something went wrong.
pub fn error_result(message: impl Into<String>) -> ToolResult {
    ToolResult {
        content: message.into(),
        is_error: true,
        metadata: Value::Null,
    }
}

pub fn success_result(content: String, metadata: Value) -> ToolResult {
    ToolResult {
        content,
        is_error: false,
        metadata,
    }
}

impl ToolDefinition {
    /// Reads a tool given from Python: either a `BuiltinTool` or a Python tool object.
    pub fn from_python(tool: &Bound<'_, PyAny>) -> PyResult<Self> {
        let approval = match tool.getattr("requires_approval") {
            Ok(requires_approval) if requires_approval.is_callable() => {
                Approval::WhenFunctionSays(requires_approval.unbind())
            }
            Ok(requires_approval) if requires_approval.is_truthy()? => Approval::Always,
            _ => Approval::Never,
        };

        if let Ok(builtin_tool) = tool.cast::<BuiltinTool>() {
            let definition = builtin_tool.get().definition;
            return Ok(ToolDefinition {
                name: definition.name.to_string(),
                description: definition.description.clone(),
                parameters: definition.parameters.clone(),
                implementation: ToolImplementation::Builtin(definition),
                approval,
            });
        }

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
                    "{} is not a tool. Use the @tool decorator, Tool(...) or BuiltinTool(...).",
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
                    move |py| {
                        function
                            .bind(py)
                            .call1((context.to_python(py, Some(call_log))?, json_to_python(py, &arguments)?))
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
            ToolImplementation::Builtin(definition) => {
                let run = definition.run;
                let arguments = Value::Object(arguments.clone());
                run_rust_tool(&self.name, move || run(arguments)).await
            }
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
            ToolImplementation::Frontend => unreachable!("the agent loop asks the app to run frontend tools"),
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
/// `invoke` returns text, or a `ToolReturn` (text plus metadata for the app).
async fn run_python_tool(
    python_tool: &Py<PyAny>,
    arguments: &Map<String, Value>,
    context: Option<(Context, SharedCallLog)>,
    python_task_locals: &TaskLocals,
) -> PyResult<ToolResult> {
    let python_tool = Python::attach(|py| python_tool.clone_ref(py));
    let arguments = Value::Object(arguments.clone());
    let tool_result = call_python(
        move |py| {
            let arguments_dict = json_to_python(py, &arguments)?;
            match context {
                Some((context, call_log)) => {
                    let ctx = context.to_python(py, Some(call_log))?;
                    python_tool.bind(py).call_method1("invoke", (arguments_dict, ctx))
                }
                None => python_tool.bind(py).call_method1("invoke", (arguments_dict,)),
            }
        },
        python_task_locals,
    )
    .await?;
    Python::attach(|py| {
        let tool_result = tool_result.bind(py);
        if let Ok(content) = tool_result.extract::<String>() {
            return Ok(success_result(content, Value::Null));
        }
        Ok(success_result(
            tool_result.getattr("content")?.extract()?,
            python_to_json(&tool_result.getattr("metadata")?)?,
        ))
    })
}
