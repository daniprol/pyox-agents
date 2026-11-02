//! Letting Python exit while Rust threads may still hand it results.
//!
//! Rust gives results to Python from its own threads: the next event of a
//! run, a tool call's result. Python 3.11 to 3.13 kill a thread that tries to
//! take the GIL while the interpreter shuts down, and a Rust thread killed in
//! the middle of a Python call crashes the process (`SIGSEGV` at exit). It
//! happens when a script ends right after its last run: the thread handing
//! over the `RunFinished` is still finishing that call.
//!
//! So every hand-over from a Rust thread is counted while it runs, and an
//! `atexit` function (registered when the module is imported) waits for the
//! running ones to finish, and stops new ones, before Python shuts down.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use pyo3_async_runtimes::generic::{ContextExt, Runtime};

/// Hand-overs to Python running on Rust threads right now.
static RUNNING: AtomicUsize = AtomicUsize::new(0);
/// Set once Python started exiting: no new hand-overs.
static PYTHON_EXITING: AtomicBool = AtomicBool::new(false);

/// How long Python's exit waits for the running hand-overs at most.
const EXIT_WAIT: Duration = Duration::from_secs(2);

/// One hand-over in progress; it ends when this value is dropped.
pub struct HandOver;

impl HandOver {
    /// Starts a hand-over, or `None` when Python is exiting (skip the call).
    pub fn start() -> Option<HandOver> {
        // Counted before checking, so `wait_at_exit` either sees it running or
        // this sees Python exiting: never neither.
        RUNNING.fetch_add(1, Ordering::SeqCst);
        if PYTHON_EXITING.load(Ordering::SeqCst) {
            RUNNING.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(HandOver)
    }
}

impl Drop for HandOver {
    fn drop(&mut self) {
        RUNNING.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Registers `wait_at_exit` with Python's `atexit`.
pub fn register(py: Python<'_>) -> PyResult<()> {
    let wait = wrap_pyfunction!(wait_at_exit, py)?;
    py.import("atexit")?.call_method1("register", (wait,))?;
    Ok(())
}

/// Called by `atexit`: stops new hand-overs, then waits (with the GIL
/// released, so they can finish) until the running ones are done.
#[pyfunction]
fn wait_at_exit(py: Python<'_>) {
    PYTHON_EXITING.store(true, Ordering::SeqCst);
    let started = Instant::now();
    py.detach(|| {
        while RUNNING.load(Ordering::SeqCst) > 0 && started.elapsed() < EXIT_WAIT {
            std::thread::sleep(Duration::from_millis(1));
        }
    });
}

/// Like `pyo3_async_runtimes::tokio::future_into_py`: a Python awaitable for
/// `future`, whose result is handed over to Python as a counted `HandOver`.
pub fn future_into_py<'py, T>(
    py: Python<'py>,
    future: impl Future<Output = PyResult<T>> + Send + 'static,
) -> PyResult<Bound<'py, PyAny>>
where
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    pyo3_async_runtimes::generic::future_into_py::<CountedTokio, _, _>(py, future)
}

/// `future_into_py`, for the event loop and contextvars in `locals`.
pub fn future_into_py_with_locals<'py, T>(
    py: Python<'py>,
    locals: TaskLocals,
    future: impl Future<Output = PyResult<T>> + Send + 'static,
) -> PyResult<Bound<'py, PyAny>>
where
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    pyo3_async_runtimes::generic::future_into_py_with_locals::<CountedTokio, _, _>(py, locals, future)
}

/// The tokio runtime pyo3-async-runtimes uses, except that the work it does
/// to give a result to Python (`spawn_blocking`) is a counted `HandOver`.
struct CountedTokio;

impl Runtime for CountedTokio {
    type JoinError = tokio::task::JoinError;
    type JoinHandle = tokio::task::JoinHandle<()>;

    fn spawn<F>(future: F) -> Self::JoinHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        pyo3_async_runtimes::tokio::get_runtime().spawn(future)
    }

    fn spawn_blocking<F>(hand_over: F) -> Self::JoinHandle
    where
        F: FnOnce() + Send + 'static,
    {
        pyo3_async_runtimes::tokio::get_runtime().spawn_blocking(move || {
            if let Some(_running) = HandOver::start() {
                hand_over();
            }
        })
    }
}

tokio::task_local! {
    /// The event loop and contextvars of the awaitable a task works for.
    static LOCALS: TaskLocals;
}

impl ContextExt for CountedTokio {
    fn scope<F, R>(locals: TaskLocals, future: F) -> Pin<Box<dyn Future<Output = R> + Send>>
    where
        F: Future<Output = R> + Send + 'static,
    {
        Box::pin(LOCALS.scope(locals, future))
    }

    fn get_task_locals() -> Option<TaskLocals> {
        LOCALS.try_with(TaskLocals::clone).ok()
    }
}
