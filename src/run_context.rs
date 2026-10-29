//! The `ctx` that tools and hooks receive: what is known about the run, plus
//! the user's own `deps` (any Python object given with `execute(deps=...)`).

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyType;
use pyox_tools::hooks::RunContext;
use pyox_tools::usage::Usage;

use crate::messages::ToolCall;

/// The context of one run, kept up to date by the agent loop.
#[derive(Clone)]
pub struct Context {
    /// The facts Rust hooks see too (conversation, turn, tokens used).
    pub run: RunContext,
    /// `Arc` lets every tool call share the object without taking the GIL.
    pub deps: Arc<Py<PyAny>>,
}

impl Context {
    /// Makes the Python `ctx` object. `tool_call` is the call a tool is running for, if any.
    pub fn to_python(&self, py: Python<'_>, tool_call: Option<&ToolCall>) -> PyResult<Py<PythonRunContext>> {
        Py::new(
            py,
            PythonRunContext {
                context: self.clone(),
                tool_call_id: tool_call.map(|call| call.id.clone()),
                tool_name: tool_call.map(|call| call.name.clone()),
            },
        )
    }
}

/// Python: `RunContext`, given as `ctx` to tools that ask for it and to hooks.
/// A read-only snapshot, made for each call.
#[pyclass(frozen, name = "RunContext", module = "pyox_agents")]
pub struct PythonRunContext {
    context: Context,
    /// The id of the tool call, inside a tool (`None` in hooks).
    #[pyo3(get)]
    tool_call_id: Option<String>,
    /// The name of the tool, inside a tool (`None` in hooks).
    #[pyo3(get)]
    tool_name: Option<String>,
}

#[pymethods]
impl PythonRunContext {
    /// Whatever was given with `execute(deps=...)`: a database pool, the user, settings...
    #[getter]
    fn deps(&self, py: Python<'_>) -> Py<PyAny> {
        self.context.deps.clone_ref(py)
    }

    #[getter]
    fn conversation_id(&self) -> Option<&str> {
        self.context.run.conversation_id.as_deref()
    }

    /// The current turn: 1 during the first model request, 2 during the second...
    #[getter]
    fn turn(&self) -> u32 {
        self.context.run.turn
    }

    /// Tokens used by the run so far.
    #[getter]
    fn usage(&self) -> Usage {
        self.context.run.usage
    }

    /// Lets Python write `RunContext[MyDeps]` in type hints; it is the same class.
    #[classmethod]
    fn __class_getitem__(class: &Bound<'_, PyType>, _deps_type: &Bound<'_, PyAny>) -> Py<PyType> {
        class.clone().unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "RunContext(conversation_id={:?}, turn={}, tool_name={:?})",
            self.context.run.conversation_id, self.context.run.turn, self.tool_name
        )
    }
}
