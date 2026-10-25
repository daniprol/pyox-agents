//! Calling Python functions (sync or async) from the agent loop.
//!
//! Used for Python tools and Python history stores.

use std::future::Future;
use std::pin::Pin;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;

/// What calling the Python function gave back.
enum CallResult {
    /// A sync function already returned its result.
    Finished(Py<PyAny>),
    /// An async function returned a coroutine, which runs on Python's event loop.
    Pending(Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>),
}

/// Makes a Python call with `call` and returns its result, awaiting it if
/// the function is async.
///
/// The call happens on a thread meant for blocking work: sync functions (like
/// compiled Rust tools) may take a while, and must not stall the network tasks.
/// `for<'py>` just means `call` works with whatever Python session it is given.
pub async fn call_python<Call>(call: Call, python_task_locals: &TaskLocals) -> PyResult<Py<PyAny>>
where
    Call: for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyAny>> + Send + 'static,
{
    let python_task_locals = python_task_locals.clone();
    let call_result = tokio::task::spawn_blocking(move || {
        // `Python::attach` gives us access to the Python interpreter (it takes the GIL).
        Python::attach(|py| -> PyResult<CallResult> {
            let result = call(py)?;
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
