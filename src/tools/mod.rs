//! Tools: what the model can call.
//!
//! A tool is implemented either in Python or in Rust:
//!
//! - **Python tools** (`pyox_agents.Tool`, usually made with `@tool`). Any object
//!   with `name`, `description`, `parameters` and an async `invoke(arguments)`
//!   method works. That includes Rust functions compiled in a separate PyO3
//!   package and wrapped with `Tool(...)`.
//! - **Built-in Rust tools** (`BuiltinTool("read_file")`). They are compiled into
//!   this library and run entirely in Rust, without touching Python.

mod builtin;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use serde_json::{Map, Value};

pub use builtin::BuiltinTool;
use builtin::BuiltinToolSpec;

use crate::json_convert::{json_to_python, python_to_json};

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
    /// A Python object with an async `invoke(arguments: dict) -> str` method.
    Python(Py<PyAny>),
    /// A tool written in Rust inside this library.
    Builtin(&'static BuiltinToolSpec),
}

/// The result of running a tool, as sent back to the model.
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    fn error(message: impl Into<String>) -> Self {
        ToolOutput {
            content: message.into(),
            is_error: true,
        }
    }
}

impl ToolDefinition {
    /// Reads a tool given from Python: either a `BuiltinTool` or a Python tool object.
    pub fn from_python(tool: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(builtin_tool) = tool.cast::<BuiltinTool>() {
            let spec = builtin_tool.get().spec;
            return Ok(ToolDefinition {
                name: spec.name.to_string(),
                description: spec.description.to_string(),
                parameters: spec.parameters(),
                implementation: ToolImplementation::Builtin(spec),
            });
        }

        if !tool.hasattr("invoke")? {
            return Err(PyValueError::new_err(format!(
                "{} is not a tool. Use the @tool decorator, Tool(...) or BuiltinTool(...).",
                tool.repr()?
            )));
        }
        Ok(ToolDefinition {
            name: tool.getattr("name")?.extract()?,
            description: tool.getattr("description")?.extract()?,
            parameters: python_to_json(&tool.getattr("parameters")?)?,
            implementation: ToolImplementation::Python(tool.clone().unbind()),
        })
    }

    /// Runs the tool. Errors never stop the agent: they are returned to the
    /// model as an error result, so it can react (for example, retry).
    pub async fn run(&self, arguments: &Value, python_task_locals: &TaskLocals) -> ToolOutput {
        let Value::Object(arguments) = arguments else {
            return ToolOutput::error(format!("Tool arguments must be a JSON object, got: {arguments}"));
        };

        match &self.implementation {
            ToolImplementation::Builtin(spec) => run_builtin_tool(spec, arguments.clone()).await,
            ToolImplementation::Python(python_tool) => {
                match run_python_tool(python_tool, arguments, python_task_locals).await {
                    Ok(content) => ToolOutput {
                        content,
                        is_error: false,
                    },
                    // `to_string()` gives e.g. "ValueError: city not found".
                    Err(python_error) => ToolOutput::error(python_error.to_string()),
                }
            }
        }
    }
}

/// Runs a Rust tool on a thread meant for blocking work, so slow tools
/// (file access, heavy computation) do not block the network tasks.
async fn run_builtin_tool(spec: &'static BuiltinToolSpec, arguments: Map<String, Value>) -> ToolOutput {
    let result = tokio::task::spawn_blocking(move || (spec.function)(&arguments)).await;
    match result {
        Ok(Ok(content)) => ToolOutput {
            content,
            is_error: false,
        },
        Ok(Err(message)) => ToolOutput::error(message),
        Err(_) => ToolOutput::error(format!("Tool {} crashed", spec.name)),
    }
}

/// Calls `await tool.invoke(arguments)` on the Python event loop and waits for it.
async fn run_python_tool(
    python_tool: &Py<PyAny>,
    arguments: &Map<String, Value>,
    python_task_locals: &TaskLocals,
) -> PyResult<String> {
    // `Python::attach` gives us access to the Python interpreter (it takes the GIL).
    // We only hold it while starting the coroutine, not while waiting for it.
    let tool_result_future = Python::attach(|py| {
        let arguments_dict = json_to_python(py, &Value::Object(arguments.clone()))?;
        let coroutine = python_tool.bind(py).call_method1("invoke", (arguments_dict,))?;
        pyo3_async_runtimes::into_future_with_locals(python_task_locals, coroutine)
    })?;

    let tool_result = tool_result_future.await?;
    Python::attach(|py| tool_result.bind(py).extract::<String>())
}
