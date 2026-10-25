//! Reading history given from Python, and history stores
//! (`Agent(history_store=...)`): where a run loads its conversation from and
//! saves its new messages to.
//!
//! A store is either:
//!
//! - **Python code**: an object with `load(conversation_id)` and optionally
//!   `save(conversation_id, new_messages_json)`, sync or async, or just a
//!   `load` function. `load` returns a list of message dicts, or JSON text
//!   (`str` or `bytes`), which is parsed in Rust without building dicts.
//! - **A Rust store** from a separately compiled package made with
//!   `pyox_tools::history`. The engine calls its Rust functions directly, so
//!   no Python runs and the GIL is never taken (see `pyox_tools::native_history`).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use pyo3_async_runtimes::TaskLocals;
use pyox_tools::native_history::NativeHistoryStore;

use crate::errors::HistoryError;
use crate::json_convert::python_to_json;
use crate::messages::Message;
use crate::python_call::call_python;

/// Reads history given from Python: a list of message dicts, or JSON text
/// (`str` or `bytes`) holding such a list.
pub fn messages_from_python(history: &Bound<'_, PyAny>) -> PyResult<Vec<Message>> {
    let parsed = if let Ok(text) = history.cast::<PyString>() {
        serde_json::from_str(text.to_str()?)
    } else if let Ok(bytes) = history.cast::<PyBytes>() {
        serde_json::from_slice(bytes.as_bytes())
    } else {
        serde_json::from_value(python_to_json(history)?)
    };
    parsed.map_err(|error| PyValueError::new_err(format!("Invalid history: {error}")))
}

pub enum HistoryStore {
    /// Python functions, sync or async. `save` is optional.
    Python { load: Py<PyAny>, save: Option<Py<PyAny>> },
    /// A Rust store, called directly. `owner` is the Python object that owns
    /// the store: holding it keeps the store (and its connection pool) alive.
    Native {
        store: NativeHistoryStore,
        #[allow(dead_code)] // never read, only held
        owner: Py<PyAny>,
    },
}

impl HistoryStore {
    pub fn from_python(store: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(native_store) = NativeHistoryStore::from_python(store)? {
            return Ok(HistoryStore::Native {
                store: native_store,
                owner: store.clone().unbind(),
            });
        }
        if store.hasattr("load")? {
            let save = match store.hasattr("save")? {
                true => Some(store.getattr("save")?.unbind()),
                false => None,
            };
            return Ok(HistoryStore::Python {
                load: store.getattr("load")?.unbind(),
                save,
            });
        }
        if store.is_callable() {
            return Ok(HistoryStore::Python {
                load: store.clone().unbind(),
                save: None,
            });
        }
        Err(PyTypeError::new_err(format!(
            "{} is not a history store: give an object with a `load(conversation_id)` method, or a function",
            store.repr()?
        )))
    }

    /// Returns the messages of a conversation, oldest first.
    pub async fn load(&self, conversation_id: &str, python_task_locals: &TaskLocals) -> PyResult<Vec<Message>> {
        let loaded = match self {
            HistoryStore::Native { store, .. } => {
                // A blocking call, on a thread meant for blocking work.
                // `pyox_tools::native_history` explains why, and how it
                // could become non-blocking later.
                let store = *store;
                let conversation_id = conversation_id.to_string();
                let messages_json = tokio::task::spawn_blocking(move || store.load(&conversation_id))
                    .await
                    .map_err(|_| HistoryError::new_err("The history store crashed"))?
                    .map_err(HistoryError::new_err)?;
                serde_json::from_str(&messages_json).map_err(|error| error.to_string())
            }
            HistoryStore::Python { load, .. } => {
                let load = Python::attach(|py| load.clone_ref(py));
                let conversation_id = conversation_id.to_string();
                let result = call_python(move |py| load.bind(py).call1((conversation_id,)), python_task_locals).await?;
                Python::attach(|py| messages_from_python(result.bind(py)).map_err(|error| error.value(py).to_string()))
            }
        };
        loaded.map_err(|error| {
            HistoryError::new_err(format!("Could not read the history of {conversation_id:?}: {error}"))
        })
    }

    /// Saves the messages a run added, after the ones already stored.
    pub async fn save(
        &self,
        conversation_id: &str,
        new_messages: &[Message],
        python_task_locals: &TaskLocals,
    ) -> PyResult<()> {
        let new_messages_json = serde_json::to_string(new_messages).expect("messages always serialize");
        let conversation_id = conversation_id.to_string();
        match self {
            HistoryStore::Native { store, .. } => {
                let store = *store;
                tokio::task::spawn_blocking(move || store.save(&conversation_id, &new_messages_json))
                    .await
                    .map_err(|_| HistoryError::new_err("The history store crashed"))?
                    .map_err(HistoryError::new_err)
            }
            HistoryStore::Python { save: Some(save), .. } => {
                let save = Python::attach(|py| save.clone_ref(py));
                call_python(
                    move |py| save.bind(py).call1((conversation_id, new_messages_json)),
                    python_task_locals,
                )
                .await?;
                Ok(())
            }
            HistoryStore::Python { save: None, .. } => Ok(()),
        }
    }
}
