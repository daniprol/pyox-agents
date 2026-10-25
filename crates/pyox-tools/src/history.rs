//! Load and save conversation history in Rust, for example from a database,
//! without going through Python.
//!
//! A history store is a struct holding what it needs (usually a database
//! connection pool), plus two async methods:
//!
//! ```ignore
//! use pyox_tools::history::{HistoryStore, RustHistoryStore, runtime};
//! use pyox_tools::messages::Message;
//!
//! struct SqliteHistory {
//!     pool: sqlx::SqlitePool, // created once, shared by every run
//! }
//!
//! impl HistoryStore for SqliteHistory {
//!     async fn load(&self, conversation_id: &str) -> Result<Vec<Message>, String> { ... }
//!     async fn save(&self, conversation_id: &str, new_messages: &[Message]) -> Result<(), String> { ... }
//! }
//!
//! /// Python: `my_package.sqlite_history("sqlite://chats.db")`
//! #[pyfunction]
//! fn sqlite_history(url: &str) -> PyResult<RustHistoryStore> {
//!     // Async code (like connecting) runs on the store's runtime.
//!     let pool = runtime().block_on(sqlx::SqlitePool::connect(url)).map_err(...)?;
//!     RustHistoryStore::new(SqliteHistory { pool })
//! }
//! ```
//!
//! In Python, give the store to the agent once and name the conversation on
//! each run:
//!
//! ```python
//! agent = Agent(model="gpt-5-mini", history_store=my_package.sqlite_history("sqlite://chats.db"))
//! await agent.run("Hi", conversation_id="user-42")
//! ```
//!
//! The agent then calls `load` before the run and `save` after it, directly
//! in Rust (see [`crate::native_history`] for how).

use std::future::Future;
use std::sync::OnceLock;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use tokio::runtime::Runtime;

use crate::messages::Message;
use crate::native_history::{self, NativeHistoryStore};

/// Implement this on a struct that loads and saves conversations.
///
/// `Send + Sync` means one store can be used from many threads at once:
/// many runs load their history at the same time. A connection pool is
/// exactly that kind of object.
pub trait HistoryStore: Send + Sync + 'static {
    /// Returns the messages of a conversation, oldest first.
    /// An unknown conversation is not an error: return an empty list.
    fn load(&self, conversation_id: &str) -> impl Future<Output = Result<Vec<Message>, String>>;

    /// Stores the messages a run added (its prompt, the answers and the tool
    /// results), after the ones already saved. Runs that fail are not saved.
    ///
    /// The default does nothing, for stores that only load.
    fn save(&self, conversation_id: &str, new_messages: &[Message]) -> impl Future<Output = Result<(), String>> {
        let _ = (conversation_id, new_messages);
        async { Ok(()) }
    }
}

/// The Tokio runtime that runs your store's async code: the queries and the
/// background tasks of your connection pool. Create your pool inside it with
/// `runtime().block_on(...)`.
///
/// Each compiled package has its own runtime (separately compiled libraries
/// can't share one), created the first time it is needed.
pub fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .thread_name("pyox-history-store")
            .enable_all()
            .build()
            .expect("could not start the history store runtime")
    })
}

/// The Python object for a Rust history store: pass it as
/// `Agent(history_store=...)`. pyox-agents finds the Rust functions through
/// `__pyox_history_store__` and calls them without Python.
#[pyclass(frozen, name = "RustHistoryStore")]
pub struct RustHistoryStore {
    /// Owns the store: it is freed when the last reference to it goes away.
    native_store: Py<PyCapsule>,
}

impl RustHistoryStore {
    /// Wraps your store so it can be returned to Python.
    pub fn new<Store: HistoryStore>(store: Store) -> PyResult<Self> {
        Python::attach(|py| {
            Ok(RustHistoryStore {
                native_store: native_history::capsule_for(py, store)?.unbind(),
            })
        })
    }
}

#[pymethods]
impl RustHistoryStore {
    /// Loads a conversation as a list of message dicts, e.g. to show it in a UI.
    fn load<'py>(&self, py: Python<'py>, conversation_id: &str) -> PyResult<Bound<'py, PyAny>> {
        let store = NativeHistoryStore::from_capsule(self.native_store.bind(py))?;
        // `detach` releases Python's global lock while the database works.
        let messages_json = py
            .detach(|| store.load(conversation_id))
            .map_err(PyRuntimeError::new_err)?;
        py.import("json")?.call_method1("loads", (messages_json,))
    }

    /// The capsule with the store's native functions (see `native_history`).
    #[getter]
    fn __pyox_history_store__(&self, py: Python<'_>) -> Py<PyCapsule> {
        self.native_store.clone_ref(py)
    }

    fn __repr__(&self) -> &'static str {
        "RustHistoryStore()"
    }
}
