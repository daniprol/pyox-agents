//! Calling Python functions (sync or async) from the agent loop.
//!
//! Used for Python tools, hooks, history stores, approval functions and
//! `on_request` handlers. `PythonCalls` says where they run:
//!
//! |                       | async run (`async for`, `await agent.run`) | sync run (`for`, `agent.run_sync`)          |
//! |-----------------------|--------------------------------------------|---------------------------------------------|
//! | sync function         | a worker thread                            | the thread reading the run (`CallerThread`) |
//! | async function        | the app's event loop                       | pyox's own event loop thread                |
//!
//! Sync functions never block an event loop: in async runs they go to worker
//! threads; in sync runs the reading thread is waiting anyway, so it runs them
//! (Django's ORM then sees the view's transaction, thread-locals just work).

use std::future::Future;
use std::pin::{Pin, pin};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PySystemExit};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3_async_runtimes::TaskLocals;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::sync::oneshot;

/// How a run calls Python: in which contextvars, and on which thread or event loop.
/// Cheap to clone (`Arc` is a shared pointer, like a Python reference).
#[derive(Clone)]
pub struct PythonCalls {
    /// The contextvars Python code sees: a copy of the app's (with the current span, when traced).
    context: Arc<Py<PyAny>>,
    /// Where async functions run. `None` in a sync run: pyox's own loop thread (started on first use).
    event_loop: Option<Arc<Py<PyAny>>>,
    /// Sync runs only: the thread reading the run, which runs the sync functions.
    caller_thread: Option<CallerThread>,
}

impl PythonCalls {
    /// For an async run: the app's running event loop, and its current contextvars.
    pub fn for_async_run(py: Python<'_>) -> PyResult<Self> {
        let event_loop = py.import("asyncio")?.call_method0("get_running_loop")?;
        Ok(PythonCalls {
            context: Arc::new(copy_context(py)?.unbind()),
            event_loop: Some(Arc::new(event_loop.unbind())),
            caller_thread: None,
        })
    }

    /// For a sync run read on this thread: no event loop, this thread's contextvars.
    pub fn for_sync_run(py: Python<'_>, caller_thread: CallerThread) -> PyResult<Self> {
        Ok(PythonCalls {
            context: Arc::new(copy_context(py)?.unbind()),
            event_loop: None,
            caller_thread: Some(caller_thread),
        })
    }

    /// The contextvars Python code runs in.
    pub fn context<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.context.bind(py).clone()
    }

    /// The same calls, in other contextvars (each traced span makes itself current this way).
    pub fn with_context(&self, context: Bound<'_, PyAny>) -> Self {
        PythonCalls {
            context: Arc::new(context.unbind()),
            ..self.clone()
        }
    }

    /// The same calls, with sync functions on a worker thread (`@tool(run_in_thread=True)`).
    pub fn on_worker_thread(&self) -> Self {
        PythonCalls {
            caller_thread: None,
            ..self.clone()
        }
    }

    /// The sync run's reading thread, if this is a sync run.
    pub fn caller_thread(&self) -> Option<&CallerThread> {
        self.caller_thread.as_ref()
    }

    /// The event loop and contextvars for awaiting Python coroutines, in the form
    /// pyo3-async-runtimes wants (`TaskLocals`).
    pub fn task_locals(&self, py: Python<'_>) -> PyResult<TaskLocals> {
        let event_loop = match &self.event_loop {
            Some(event_loop) => event_loop.bind(py).clone(),
            None => py.import("pyox_agents._background_loop")?.call_method0("get")?,
        };
        Ok(TaskLocals::new(event_loop).with_context(self.context(py)))
    }
}

/// `contextvars.copy_context()`.
fn copy_context(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("contextvars")?.call_method0("copy_context")
}

/// What calling the Python function gave back.
enum CallResult {
    /// A sync function already returned its result.
    Finished(Py<PyAny>),
    /// An async function returned a coroutine, which runs on an event loop.
    Pending(Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>),
}

/// Calls `function(*arguments)` and returns its result, awaiting it if the
/// function is async. `make_arguments` builds the arguments once the call
/// has access to Python.
///
/// The function runs in a copy of the run's context, like
/// `contextvars.copy_context().run(function, *arguments)`: it sees the app's
/// context variables (a request id, the current tracing span...). A copy,
/// because two threads cannot enter the same context at once.
///
/// `for<'py>` just means `make_arguments` works with whatever Python session it is given.
pub async fn call_python<MakeArguments>(
    function: Py<PyAny>,
    make_arguments: MakeArguments,
    python_calls: &PythonCalls,
) -> PyResult<Py<PyAny>>
where
    MakeArguments: for<'py> FnOnce(Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> + Send + 'static,
{
    let python_calls = python_calls.clone();
    let caller_thread = python_calls.caller_thread.clone();
    let call = move || {
        // `Python::attach` gives us access to the Python interpreter (it takes the GIL).
        Python::attach(|py| -> PyResult<CallResult> {
            let mut arguments = vec![function.into_bound(py)];
            arguments.extend(make_arguments(py)?);
            let context = python_calls.context(py).call_method0("copy")?;
            let result = context.call_method1("run", PyTuple::new(py, arguments)?)?;
            if !result.hasattr("__await__")? {
                return Ok(CallResult::Finished(result.unbind()));
            }
            // Run it through `Cancellable`, so dropping the future (a stopped
            // or cancelled run) cancels the Python task too.
            let cancellable = py
                .import("pyox_agents._tasks")?
                .getattr("Cancellable")?
                .call1((result,))?;
            let task_locals = python_calls.task_locals(py)?;
            let future = pyo3_async_runtimes::into_future_with_locals(&task_locals, cancellable.call_method0("run")?)?;
            let mut cancel_on_drop = CancelOnDrop(Some(cancellable.unbind()));
            Ok(CallResult::Pending(Box::pin(async move {
                let result = future.await;
                cancel_on_drop.disarm(); // finished: nothing to cancel
                result
            })))
        })
    };

    // A sync run: the call goes to the thread reading it. Otherwise: a worker
    // thread (one meant for blocking work, so the network tasks never stall).
    let call_result = match caller_thread {
        Some(caller_thread) => caller_thread.run(call).await?,
        None => tokio::task::spawn_blocking(call)
            .await
            .map_err(|_| PyRuntimeError::new_err("The Python call crashed"))?,
    }?;

    match call_result {
        CallResult::Finished(result) => Ok(result),
        CallResult::Pending(future) => future.await,
    }
}

/// Holds a `pyox_agents._tasks.Cancellable` while its task runs. When the
/// future awaiting it is dropped before the end, `drop` cancels the task.
/// (`Drop` is like `__del__`, but runs at a known moment: when the value goes away.)
struct CancelOnDrop(Option<Py<PyAny>>);

impl CancelOnDrop {
    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(cancellable) = self.0.take() {
            Python::attach(|py| {
                // `let _ =`: if cancelling fails there is nothing more to do.
                let _ = cancellable.bind(py).call_method0("cancel");
            });
        }
    }
}

/// A Python call waiting for the thread that reads a sync run.
type PythonJob = Box<dyn FnOnce() + Send>;

/// The thread that reads a sync run, seen as a place to run Python: the agent
/// loop sends it jobs, and the thread runs them while it waits for the run's
/// next event (see `wait`).
#[derive(Clone)]
pub struct CallerThread {
    sender: UnboundedSender<PythonJob>,
    /// Shared, so a wait inside a job (`ctx.ask_sync`) can run the next jobs too.
    /// (`tokio::sync::Mutex` is a lock that can be held while waiting.)
    jobs: Arc<tokio::sync::Mutex<UnboundedReceiver<PythonJob>>>,
    /// A `KeyboardInterrupt` (or `SystemExit`) raised inside a job: it goes to
    /// the app, not to the model as a tool error.
    interrupted: Arc<Mutex<Option<PyErr>>>,
}

impl CallerThread {
    pub fn new() -> Self {
        let (sender, receiver) = unbounded_channel();
        CallerThread {
            sender,
            jobs: Arc::new(tokio::sync::Mutex::new(receiver)),
            interrupted: Arc::new(Mutex::new(None)),
        }
    }

    /// Runs `call` on the reading thread and waits for its result.
    async fn run<T: Send + 'static>(
        &self,
        call: impl FnOnce() -> PyResult<T> + Send + 'static,
    ) -> PyResult<PyResult<T>> {
        let (result_sender, result) = oneshot::channel();
        let interrupted = self.interrupted.clone();
        let job: PythonJob = Box::new(move || {
            let outcome = call();
            if let Err(error) = &outcome {
                Python::attach(|py| {
                    if error.is_instance_of::<PyKeyboardInterrupt>(py) || error.is_instance_of::<PySystemExit>(py) {
                        *interrupted.lock().unwrap() = Some(error.clone_ref(py));
                    }
                });
            }
            let _ = result_sender.send(outcome); // `_ =`: nobody waits any more if the run was aborted
        });
        let closed = || PyRuntimeError::new_err("The sync run was closed");
        self.sender.send(job).map_err(|_| closed())?;
        result.await.map_err(|_| closed())
    }
}

/// What woke a `wait` up.
enum Wake<T> {
    Done(T),
    Job(PythonJob),
    Tick,
}

/// Waits on this thread for `future` (which runs on the tokio runtime), with
/// the GIL released so the run (and other Python threads) can use Python.
///
/// Meanwhile it runs the Python calls the sync run sends to `caller_thread`,
/// and every 100 ms lets Python handle Ctrl+C: `check_signals` raises
/// `KeyboardInterrupt` (it does nothing off the main thread, as in Python).
pub fn wait<T: Send>(
    py: Python<'_>,
    caller_thread: Option<&CallerThread>,
    future: impl Future<Output = T> + Send,
) -> PyResult<T> {
    let runtime = pyo3_async_runtimes::tokio::get_runtime();
    let mut future = pin!(future);
    loop {
        // `py.detach`: release the GIL while waiting.
        let wake = py.detach(|| {
            runtime.block_on(async {
                let next_job = async {
                    match caller_thread {
                        Some(caller_thread) => caller_thread.jobs.lock().await.recv().await,
                        None => std::future::pending().await, // no jobs to run: wait forever
                    }
                };
                tokio::select! {
                    biased; // check in this order
                    Some(job) = next_job => Wake::Job(job),
                    output = &mut future => Wake::Done(output),
                    _ = tokio::time::sleep(Duration::from_millis(100)) => Wake::Tick,
                }
            })
        });
        match wake {
            Wake::Done(output) => return Ok(output),
            Wake::Job(job) => {
                job();
                if let Some(caller_thread) = caller_thread
                    && let Some(interrupt) = caller_thread.interrupted.lock().unwrap().take()
                {
                    return Err(interrupt);
                }
            }
            Wake::Tick => py.check_signals()?,
        }
    }
}

/// Raises a clear error when this process was forked after pyox-agents ran in
/// its parent (Gunicorn `--preload`, Celery prefork, `multiprocessing`). The
/// runtime's threads did not survive the fork, so a run would hang forever.
pub fn check_not_forked() -> PyResult<()> {
    // `OnceLock`: set once, by the first run in this process.
    static FIRST_RUN_PID: OnceLock<u32> = OnceLock::new();
    if *FIRST_RUN_PID.get_or_init(std::process::id) == std::process::id() {
        return Ok(());
    }
    Err(PyRuntimeError::new_err(
        "pyox-agents ran in the parent process before this process was forked, and cannot run here. \
         Run agents only after the fork: in Gunicorn, without --preload (or not in the master); \
         in Celery, inside the tasks.",
    ))
}
