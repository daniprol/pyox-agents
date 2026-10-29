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

mod builtin;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use pyox_tools::native::NativeTool;
use serde_json::{Map, Value};

pub use builtin::BuiltinTool;
/// The result of running a tool, as sent back to the model.
pub use pyox_tools::hooks::ToolResult;

use crate::json_convert::{json_to_python, python_to_json};
use crate::messages::ToolCall;
use crate::python_call::call_python;
use crate::run_context::Context;

/// Everything the agent loop needs to know about one tool.
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON schema describing the arguments.
    pub parameters: Value,
    pub implementation: ToolImplementation,
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
}

/// A tool result telling the model something went wrong.
pub fn error_result(message: impl Into<String>) -> ToolResult {
    ToolResult {
        content: message.into(),
        is_error: true,
    }
}

fn success_result(content: String) -> ToolResult {
    ToolResult {
        content,
        is_error: false,
    }
}

impl ToolDefinition {
    /// Reads a tool given from Python: either a `BuiltinTool` or a Python tool object.
    pub fn from_python(tool: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(builtin_tool) = tool.cast::<BuiltinTool>() {
            let definition = builtin_tool.get().definition;
            return Ok(ToolDefinition {
                name: definition.name.to_string(),
                description: definition.description.clone(),
                parameters: definition.parameters.clone(),
                implementation: ToolImplementation::Builtin(definition),
            });
        }

        let implementation = match NativeTool::from_python(tool)? {
            Some(native_tool) => ToolImplementation::Native(native_tool),
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
        })
    }

    /// Runs the tool. Errors never stop the agent: they are returned to the
    /// model as an error result, so it can react (for example, retry).
    pub async fn run(&self, call: &ToolCall, context: &Context, python_task_locals: &TaskLocals) -> ToolResult {
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
                let context = takes_context.then(|| (context.clone(), call.clone()));
                match run_python_tool(tool, arguments, context, python_task_locals).await {
                    Ok(content) => success_result(content),
                    // `to_string()` gives e.g. "ValueError: city not found".
                    Err(python_error) => error_result(python_error.to_string()),
                }
            }
        }
    }
}

/// Runs a Rust tool on a thread meant for blocking work, so slow tools
/// (file access, heavy computation) do not block the network tasks.
/// No Python is involved.
async fn run_rust_tool(tool_name: &str, run: impl FnOnce() -> Result<String, String> + Send + 'static) -> ToolResult {
    match tokio::task::spawn_blocking(run).await {
        Ok(Ok(content)) => success_result(content),
        Ok(Err(message)) => error_result(message),
        Err(_) => error_result(format!("Tool {tool_name} crashed")),
    }
}

/// Calls `tool.invoke(arguments)`, or `tool.invoke(arguments, ctx)` when a
/// context is given, awaiting the result if the tool is async.
async fn run_python_tool(
    python_tool: &Py<PyAny>,
    arguments: &Map<String, Value>,
    context: Option<(Context, ToolCall)>,
    python_task_locals: &TaskLocals,
) -> PyResult<String> {
    let python_tool = Python::attach(|py| python_tool.clone_ref(py));
    let arguments = Value::Object(arguments.clone());
    let tool_result = call_python(
        move |py| {
            let arguments_dict = json_to_python(py, &arguments)?;
            match context {
                Some((context, call)) => {
                    let ctx = context.to_python(py, Some(&call))?;
                    python_tool.bind(py).call_method1("invoke", (arguments_dict, ctx))
                }
                None => python_tool.bind(py).call_method1("invoke", (arguments_dict,)),
            }
        },
        python_task_locals,
    )
    .await?;
    Python::attach(|py| tool_result.bind(py).extract::<String>())
}
