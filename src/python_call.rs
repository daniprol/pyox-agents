//! Calling Python functions (sync or async) from the agent loop.
//!
//! Used for Python tools, hooks, history stores, approval functions and
//! `on_request` handlers.

use std::future::Future;
use std::pin::Pin;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3_async_runtimes::TaskLocals;

/// What calling the Python function gave back.
enum CallResult {
    /// A sync function already returned its result.
    Finished(Py<PyAny>),
    /// An async function returned a coroutine, which runs on Python's event loop.
    Pending(Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>),
}

/// Calls `function(*arguments)` and returns its result, awaiting it if the
/// function is async. `make_arguments` builds the arguments once the call
/// has access to Python.
///
/// The call happens on a thread meant for blocking work: sync functions (like
/// compiled Rust tools) may take a while, and must not stall the network tasks.
///
/// That thread has no contextvars of its own, so the function runs in a copy
/// of the run's context, like `contextvars.copy_context().run(function, *arguments)`:
/// a sync hook sees the app's context variables (a request id, the current
/// tracing span...), exactly like an async one. A copy, because two threads
/// cannot enter the same context at once.
///
/// `for<'py>` just means `make_arguments` works with whatever Python session it is given.
pub async fn call_python<MakeArguments>(
    function: Py<PyAny>,
    make_arguments: MakeArguments,
    python_task_locals: &TaskLocals,
) -> PyResult<Py<PyAny>>
where
    MakeArguments: for<'py> FnOnce(Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> + Send + 'static,
{
    let python_task_locals = python_task_locals.clone();
    let call_result = tokio::task::spawn_blocking(move || {
        // `Python::attach` gives us access to the Python interpreter (it takes the GIL).
        Python::attach(|py| -> PyResult<CallResult> {
            let mut arguments = vec![function.into_bound(py)];
            arguments.extend(make_arguments(py)?);
            let context = python_task_locals.context(py).call_method0("copy")?;
            let result = context.call_method1("run", PyTuple::new(py, arguments)?)?;
            if result.hasattr("__await__")? {
                let future = pyo3_async_runtimes::into_future_with_locals(&python_task_locals, result)?;
                Ok(CallResult::Pending(Box::pin(future)))
            } else {
                Ok(CallResult::Finished(result.unbind()))
            }
        })
    })
    .await
    .map_err(|_| PyRuntimeError::new_err("The Python call crashed"))??;

    match call_result {
        CallResult::Finished(result) => Ok(result),
        CallResult::Pending(future) => future.await,
    }
}
