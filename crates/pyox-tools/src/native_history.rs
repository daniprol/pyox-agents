//! Calling a history store's Rust code directly, without going through Python.
//!
//! This works like native tools (see [`crate::native`]): the store's package
//! and pyox-agents are separately compiled libraries, so they talk through a
//! small C-style interface stored in a Python capsule:
//!
//! - `NativeHistoryFunctions`: a pointer to your store, plus `load`, `save`
//!   and `free_result` function pointers, and a version number.
//! - The engine reads the capsule once, when the `Agent` is created, and calls
//!   `load` / `save` directly on each run. Messages cross the boundary as JSON
//!   text (a list of messages): Rust types have no stable memory layout
//!   between separately compiled libraries, and JSON parsing in Rust is fast.
//!
//! The calls are *blocking*: `load` waits until your async method is done,
//! running it on the store's own Tokio runtime (see [`crate::history::runtime`]).
//! The engine makes the call on a thread meant for blocking work, so it never
//! stalls the agent loop. A Rust future can't be passed instead, because each
//! library has its own copy of Tokio, and a future made by one can't safely be
//! run by the other.
//!
//! Possible future improvement: make the calls non-blocking with a callback.
//! `load` would start the work on the store's runtime and return at once; when
//! it finishes, the store calls a function pointer given by the engine, which
//! wakes the waiting run (e.g. through a `tokio::sync::oneshot` channel). That
//! frees the thread that now waits for the database. Only this file and the
//! engine's `history_store.rs` would change: the `HistoryStore` trait users
//! implement stays the same.
//!
//! `unsafe` marks code where Rust can't check memory safety by itself; the
//! comments next to each `unsafe` block explain why it is correct.

use std::ffi::{CStr, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::history::{HistoryStore, runtime};
use crate::messages::Message;
use crate::native::{NativeResult, free_result};

/// Bump when `NativeHistoryFunctions` (or `NativeResult`) change.
pub const INTERFACE_VERSION: u32 = 1;

/// The capsule name; checked when reading, so a wrong capsule is rejected.
pub const CAPSULE_NAME: &CStr = c"pyox_tools.native_history_store.v1";

/// The Python attribute that holds the capsule on a history store object.
pub const CAPSULE_ATTRIBUTE: &str = "__pyox_history_store__";

/// The functions of one history store.
/// `#[repr(C)]` gives the struct the same memory layout in every library.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NativeHistoryFunctions {
    pub interface_version: u32,
    /// The store, as an untyped pointer: only `load` and `save` know its type.
    pub store: *const c_void,
    /// Loads a conversation. Returns the JSON text of its list of messages.
    pub load: unsafe extern "C" fn(
        store: *const c_void,
        conversation_id_pointer: *const u8,
        conversation_id_length: usize,
    ) -> NativeResult,
    /// Saves new messages, given as the JSON text of a list. Returns empty text.
    pub save: unsafe extern "C" fn(
        store: *const c_void,
        conversation_id_pointer: *const u8,
        conversation_id_length: usize,
        messages_pointer: *const u8,
        messages_length: usize,
    ) -> NativeResult,
    /// Frees a result returned by `load` or `save`.
    pub free_result: unsafe extern "C" fn(result: NativeResult),
}

// SAFETY: `store` points to a `HistoryStore`, which must be `Send + Sync`, so
// it can be moved to and used from any thread. The function pointers are
// plain code. Rust can't see that through a raw pointer, so we say it here.
unsafe impl Send for NativeHistoryFunctions {}
unsafe impl Sync for NativeHistoryFunctions {}

// ---------------------------------------------------------------------------
// Store package side: export a `HistoryStore` as native functions
// ---------------------------------------------------------------------------

/// Makes the capsule for a store. The capsule owns the store and frees it
/// when Python frees the capsule.
pub(crate) fn capsule_for<Store: HistoryStore>(py: Python<'_>, store: Store) -> PyResult<Bound<'_, PyCapsule>> {
    let functions = functions_for(Box::new(store));
    PyCapsule::new_with_value_and_destructor(py, functions, CAPSULE_NAME, |functions, _| {
        // SAFETY: `functions.store` was made by `functions_for::<Store>` and
        // the capsule (the only owner) is being destroyed.
        unsafe { free_store::<Store>(functions.store) }
    })
}

fn functions_for<Store: HistoryStore>(store: Box<Store>) -> NativeHistoryFunctions {
    NativeHistoryFunctions {
        interface_version: INTERFACE_VERSION,
        // `Box::into_raw` gives up ownership: `free_store` takes it back.
        store: Box::into_raw(store) as *const c_void,
        load: load_native::<Store>,
        save: save_native::<Store>,
        free_result,
    }
}

/// # Safety
/// `store` must come from `functions_for::<Store>` and be freed only once.
unsafe fn free_store<Store: HistoryStore>(store: *const c_void) {
    drop(unsafe { Box::from_raw(store as *mut Store) });
}

unsafe extern "C" fn load_native<Store: HistoryStore>(
    store: *const c_void,
    conversation_id_pointer: *const u8,
    conversation_id_length: usize,
) -> NativeResult {
    // SAFETY: the engine passes the store pointer from this same capsule
    // (kept alive while in use) and a string it owns during the call.
    let (store, conversation_id) = unsafe {
        (
            &*(store as *const Store),
            text_from_parts(conversation_id_pointer, conversation_id_length),
        )
    };
    run_guarded(|| {
        let messages = runtime().block_on(store.load(conversation_id))?;
        Ok(serde_json::to_string(&messages).expect("messages always serialize"))
    })
}

unsafe extern "C" fn save_native<Store: HistoryStore>(
    store: *const c_void,
    conversation_id_pointer: *const u8,
    conversation_id_length: usize,
    messages_pointer: *const u8,
    messages_length: usize,
) -> NativeResult {
    // SAFETY: as in `load_native`; the messages text is owned by the engine too.
    let (store, conversation_id, messages_json) = unsafe {
        (
            &*(store as *const Store),
            text_from_parts(conversation_id_pointer, conversation_id_length),
            text_from_parts(messages_pointer, messages_length),
        )
    };
    run_guarded(|| {
        let new_messages: Vec<Message> =
            serde_json::from_str(messages_json).map_err(|error| format!("Invalid messages: {error}"))?;
        runtime().block_on(store.save(conversation_id, &new_messages))?;
        Ok(String::new())
    })
}

/// Rebuilds a `&str` from a pointer and a length.
///
/// # Safety
/// The parts must describe valid UTF-8 text that outlives the returned `&str`.
unsafe fn text_from_parts<'a>(pointer: *const u8, length: usize) -> &'a str {
    unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(pointer, length)) }
}

/// Runs the store's code, turning a panic into an error: a panic must never
/// cross into another library.
fn run_guarded(work: impl FnOnce() -> Result<String, String>) -> NativeResult {
    let outcome = catch_unwind(AssertUnwindSafe(work));
    NativeResult::new(outcome.unwrap_or_else(|_| Err("The history store crashed".to_string())))
}

// ---------------------------------------------------------------------------
// Engine side: read the capsule and call the store
// ---------------------------------------------------------------------------

/// A history store the engine can call directly.
///
/// It only holds pointers: whoever uses it must keep the Python store object
/// (which owns the capsule) alive.
#[derive(Clone, Copy)]
pub struct NativeHistoryStore {
    functions: NativeHistoryFunctions,
}

impl NativeHistoryStore {
    /// Reads the native functions of a Python object, if it has them.
    pub fn from_python(store: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let Ok(capsule) = store.getattr(CAPSULE_ATTRIBUTE) else {
            return Ok(None);
        };
        Ok(Some(Self::from_capsule(capsule.cast::<PyCapsule>()?)?))
    }

    pub(crate) fn from_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        let pointer = capsule.pointer_checked(Some(CAPSULE_NAME))?;
        // SAFETY: the capsule name guarantees it holds a `NativeHistoryFunctions`
        // (created by `capsule_for`). We copy it out right away.
        let functions = unsafe { *pointer.cast::<NativeHistoryFunctions>().as_ref() };
        if functions.interface_version != INTERFACE_VERSION {
            return Err(PyValueError::new_err(format!(
                "History store was built with native interface version {}, but this pyox-agents uses version {}",
                functions.interface_version, INTERFACE_VERSION
            )));
        }
        Ok(NativeHistoryStore { functions })
    }

    /// Loads a conversation: `Ok(json)` is the JSON text of a list of messages.
    /// Blocks until the store is done.
    pub fn load(&self, conversation_id: &str) -> Result<String, String> {
        // SAFETY: the functions come from a checked capsule that the caller
        // keeps alive. The id stays alive during the call, and the result is
        // freed once, by the library that allocated it.
        unsafe {
            let result = (self.functions.load)(self.functions.store, conversation_id.as_ptr(), conversation_id.len());
            result.read_and_free(self.functions.free_result)
        }
    }

    /// Saves new messages, given as the JSON text of a list. Blocks until the store is done.
    pub fn save(&self, conversation_id: &str, new_messages_json: &str) -> Result<(), String> {
        // SAFETY: as in `load`.
        unsafe {
            let result = (self.functions.save)(
                self.functions.store,
                conversation_id.as_ptr(),
                conversation_id.len(),
                new_messages_json.as_ptr(),
                new_messages_json.len(),
            );
            result.read_and_free(self.functions.free_result).map(|_| ())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;

    /// Keeps conversations in memory. Fails on purpose for some ids.
    #[derive(Default)]
    struct MemoryHistory {
        conversations: Mutex<HashMap<String, Vec<Message>>>,
    }

    impl HistoryStore for MemoryHistory {
        async fn load(&self, conversation_id: &str) -> Result<Vec<Message>, String> {
            match conversation_id {
                "broken" => Err("Database is down".into()),
                "bug" => panic!("simulated bug"),
                _ => Ok(self
                    .conversations
                    .lock()
                    .unwrap()
                    .get(conversation_id)
                    .cloned()
                    .unwrap_or_default()),
            }
        }

        async fn save(&self, conversation_id: &str, new_messages: &[Message]) -> Result<(), String> {
            let mut conversations = self.conversations.lock().unwrap();
            let conversation = conversations.entry(conversation_id.to_string()).or_default();
            conversation.extend_from_slice(new_messages);
            Ok(())
        }
    }

    #[test]
    fn messages_errors_and_panics_cross_the_interface() {
        let functions = functions_for(Box::new(MemoryHistory::default()));
        let store = NativeHistoryStore { functions };

        assert_eq!(store.load("new"), Ok("[]".into()));
        let messages_json = r#"[{"role":"user","content":"Hi"}]"#;
        assert_eq!(store.save("chat", messages_json), Ok(()));
        assert_eq!(store.save("chat", messages_json), Ok(()));
        assert_eq!(
            store.load("chat"),
            Ok(r#"[{"role":"user","content":"Hi"},{"role":"user","content":"Hi"}]"#.into())
        );

        assert_eq!(store.load("broken"), Err("Database is down".into()));
        assert_eq!(store.load("bug"), Err("The history store crashed".into()));
        assert!(
            store
                .save("chat", "not json")
                .unwrap_err()
                .starts_with("Invalid messages")
        );

        // SAFETY: made by `functions_for::<MemoryHistory>` above, freed once.
        unsafe { free_store::<MemoryHistory>(functions.store) };
    }
}
