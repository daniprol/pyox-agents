//! The `ctx` that tools and hooks receive: what is known about the run, plus
//! the user's own `deps` (any Python object given with `execute(deps=...)`).
//!
//! Inside a tool, `ctx` can also wait for a person (`ctx.ask`) and save work
//! that must not be repeated when the call runs again (`ctx.step`); see
//! `human_input`.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};
use pyo3_async_runtimes::tokio::{future_into_py_with_locals, get_current_locals};
use pyox_tools::hooks::RunContext;
use pyox_tools::usage::Usage;
use serde_json::Value;

use crate::errors::StopToolCall;
use crate::human_input::{HumanInput, SharedCallLog, new_request, reply_to, reply_to_python};
use crate::json_convert::{json_to_python, python_to_json};
use crate::messages::{Reply, RequestKind};
use crate::python_call::call_python;

/// The context of one run, kept up to date by the agent loop.
#[derive(Clone)]
pub struct Context {
    /// The facts Rust hooks see too (conversation, turn, tokens used).
    pub run: RunContext,
    /// `Arc` lets every tool call share the object without taking the GIL.
    pub deps: Arc<Py<PyAny>>,
    /// Where tool calls send requests for a person (see `human_input`).
    pub human_input: Arc<HumanInput>,
}

impl Context {
    /// Makes the Python `ctx` object. `call_log` is the tool call it is made
    /// for (`None` in hooks).
    pub fn to_python(&self, py: Python<'_>, call_log: Option<SharedCallLog>) -> PyResult<Py<PythonRunContext>> {
        let call = call_log.as_ref().map(|log| log.lock().unwrap().call.clone());
        Py::new(
            py,
            PythonRunContext {
                context: self.clone(),
                tool_call_id: call.as_ref().map(|call| call.id.clone()),
                tool_name: call.map(|call| call.name),
                call_log,
            },
        )
    }
}

/// Python: `RunContext`, given as `ctx` to tools that ask for it and to hooks.
/// Made for each call.
#[pyclass(frozen, name = "RunContext", module = "pyox_agents")]
pub struct PythonRunContext {
    context: Context,
    /// The id of the tool call, inside a tool (`None` in hooks).
    #[pyo3(get)]
    tool_call_id: Option<String>,
    /// The name of the tool, inside a tool (`None` in hooks).
    #[pyo3(get)]
    tool_name: Option<String>,
    /// What this tool call asked so far (`None` in hooks).
    call_log: Option<SharedCallLog>,
}

impl PythonRunContext {
    fn call_log(&self, method: &str) -> PyResult<SharedCallLog> {
        self.call_log
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err(format!("ctx.{method} only works inside a tool")))
    }
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

    /// Asks a person something and waits for the reply: `answer = await ctx.ask(data, metadata=...)`.
    ///
    /// Returns an `Answer`. `data` is what to show; `metadata` is for the app
    /// only and never reaches the model. Give a `key` to match the answer by
    /// name instead of by position (useful when the tool asks different
    /// questions depending on earlier answers).
    ///
    /// With a live `on_request` handler the tool simply goes on. Otherwise the
    /// run stops here, and the call runs again from its start once answered;
    /// this `ask` then returns the stored answer at once.
    #[pyo3(signature = (data = None, *, metadata = None, key = None))]
    fn ask<'py>(
        &self,
        py: Python<'py>,
        data: Option<Bound<'py, PyAny>>,
        metadata: Option<Bound<'py, PyAny>>,
        key: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let log = self.call_log("ask")?;
        let mut request = new_request(RequestKind::Input, optional_json(data)?, optional_json(metadata)?);
        request.key = key;
        let human_input = self.context.human_input.clone();
        let python_task_locals = get_current_locals(py)?;

        // Returns a Python awaitable that runs this Rust `async` block.
        future_into_py_with_locals(py, python_task_locals.clone(), async move {
            match reply_to(&log, request, &human_input, &python_task_locals).await? {
                Some(reply @ Reply::Answer { .. }) => Python::attach(|py| Ok(reply_to_python(py, &reply)?.unbind())),
                Some(Reply::Deny { message, .. }) => {
                    // The call ends with the denial as its result.
                    log.lock().unwrap().denied = Some(message.clone());
                    Err(StopToolCall::new_err(message))
                }
                Some(Reply::Approve { .. }) => Err(PyTypeError::new_err(
                    "Reply to a question with Answer(...) or Deny(...), not Approve(...)",
                )),
                // Nobody answered: stop the tool here. The agent loop sees the
                // unanswered request and records the call as waiting.
                None => Err(StopToolCall::new_err("Waiting for an answer")),
            }
        })
    }

    /// Runs `function(*args, **kwargs)` once and saves its result:
    /// `flights = await ctx.step("search", search_flights, city)`.
    ///
    /// When the call runs again after a stop, the step returns the saved
    /// result instead of running again. Use it for work before a `ctx.ask`
    /// that is slow, random, or changes something. The result must be JSON
    /// (it is saved in the history), and is returned as JSON gives it back
    /// (e.g. tuples become lists), the first time too. `function` may be async.
    #[pyo3(signature = (name, function, *args, **kwargs))]
    fn step<'py>(
        &self,
        py: Python<'py>,
        name: String,
        function: Py<PyAny>,
        args: Py<PyTuple>,
        kwargs: Option<Py<PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let log = self.call_log("step")?;
        let saved = log.lock().unwrap().start_step(&name)?;
        let python_task_locals = get_current_locals(py)?;

        future_into_py_with_locals(py, python_task_locals.clone(), async move {
            let result = match saved {
                Some(result) => result,
                None => {
                    let result = call_python(
                        move |py| {
                            function
                                .bind(py)
                                .call(args.bind(py), kwargs.as_ref().map(|kwargs| kwargs.bind(py)))
                        },
                        &python_task_locals,
                    )
                    .await?;
                    let result = Python::attach(|py| python_to_json(result.bind(py)))?;
                    log.lock().unwrap().steps.insert(name, result.clone());
                    result
                }
            };
            Python::attach(|py| Ok(json_to_python(py, &result)?.unbind()))
        })
    }

    /// Inside a tool that needed approval: the `Approve` reply (with its
    /// `metadata`). `None` otherwise.
    #[getter]
    fn approval<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(log) = &self.call_log else {
            return Ok(None);
        };
        match log.lock().unwrap().approval() {
            Some(reply) => Ok(Some(reply_to_python(py, reply)?)),
            None => Ok(None),
        }
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

/// `None` becomes JSON `null`.
fn optional_json(value: Option<Bound<'_, PyAny>>) -> PyResult<Value> {
    match value {
        Some(value) => python_to_json(&value),
        None => Ok(Value::Null),
    }
}
