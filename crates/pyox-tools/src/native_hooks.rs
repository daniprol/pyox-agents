//! Calling Rust hooks directly, without going through Python.
//!
//! This works like native history stores (see [`crate::native_history`]):
//! the hooks' package and pyox-agents are separately compiled libraries, so
//! they talk through a small C-style interface stored in a Python capsule:
//!
//! - `NativeHooksFunctions`: a pointer to your `Hooks`, a `call` function
//!   pointer that runs one of them, which hooks are registered, and a version.
//! - The engine reads the capsule once, when the `Agent` is created, and calls
//!   `call` directly at each hook point. The context and the hook's arguments
//!   cross the boundary as JSON text, like messages do for history stores.
//!
//! `unsafe` marks code where Rust can't check memory safety by itself; the
//! comments next to each `unsafe` block explain why it is correct.

use std::ffi::{CStr, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use serde_json::Value;

use crate::hooks::{Hooks, RunContext};
use crate::native::{NativeResult, free_result};

/// Bump when `NativeHooksFunctions`, `Hook` or the hook arguments change.
pub const INTERFACE_VERSION: u32 = 1;

/// The capsule name; checked when reading, so a wrong capsule is rejected.
pub const CAPSULE_NAME: &CStr = c"pyox_tools.native_hooks.v1";

/// The Python attribute that holds the capsule on a hooks object.
pub const CAPSULE_ATTRIBUTE: &str = "__pyox_hooks__";

/// The points of a run where hooks are called. Crosses the interface as a
/// number (`hook as u32`), so the order of the variants must never change.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hook {
    BeforeRun = 0,
    BeforeModelRequest = 1,
    AfterModelResponse = 2,
    BeforeToolCall = 3,
    AfterToolCall = 4,
    AfterRun = 5,
}

impl Hook {
    pub const ALL: [Hook; 6] = [
        Hook::BeforeRun,
        Hook::BeforeModelRequest,
        Hook::AfterModelResponse,
        Hook::BeforeToolCall,
        Hook::AfterToolCall,
        Hook::AfterRun,
    ];

    /// The name of the hook, which is also the name of the Python method.
    pub fn name(self) -> &'static str {
        match self {
            Hook::BeforeRun => "before_run",
            Hook::BeforeModelRequest => "before_model_request",
            Hook::AfterModelResponse => "after_model_response",
            Hook::BeforeToolCall => "before_tool_call",
            Hook::AfterToolCall => "after_tool_call",
            Hook::AfterRun => "after_run",
        }
    }

    fn from_number(number: u32) -> Option<Hook> {
        Hook::ALL.into_iter().find(|hook| *hook as u32 == number)
    }

    /// This hook's bit in `NativeHooksFunctions::registered_hooks`.
    fn bit(self) -> u32 {
        1 << (self as u32)
    }
}

/// The functions of one set of hooks.
/// `#[repr(C)]` gives the struct the same memory layout in every library.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NativeHooksFunctions {
    pub interface_version: u32,
    /// The `Hooks`, as an untyped pointer: only `call` knows its type.
    pub hooks: *const c_void,
    /// One bit per `Hook` (see `Hook::bit`): set when that hook is registered.
    pub registered_hooks: u32,
    /// Runs one hook. `context` is the JSON of a `RunContext`, `arguments`
    /// the JSON list of the hook's arguments. Returns the JSON of what the
    /// hook may change, or empty text.
    pub call: unsafe extern "C" fn(
        hooks: *const c_void,
        hook: u32,
        context_pointer: *const u8,
        context_length: usize,
        arguments_pointer: *const u8,
        arguments_length: usize,
    ) -> NativeResult,
    /// Frees a result returned by `call`.
    pub free_result: unsafe extern "C" fn(result: NativeResult),
}

// SAFETY: `hooks` points to a `Hooks`, whose closures are all `Send + Sync`,
// so it can be used from any thread. Rust can't see that through a raw
// pointer, so we say it here.
unsafe impl Send for NativeHooksFunctions {}
unsafe impl Sync for NativeHooksFunctions {}

// ---------------------------------------------------------------------------
// Hooks package side: export `Hooks` as native functions
// ---------------------------------------------------------------------------

/// Makes the capsule for a set of hooks. The capsule owns the hooks and
/// frees them when Python frees the capsule.
pub(crate) fn capsule_for(py: Python<'_>, hooks: Hooks) -> PyResult<Bound<'_, PyCapsule>> {
    let functions = functions_for(Box::new(hooks));
    PyCapsule::new_with_value_and_destructor(py, functions, CAPSULE_NAME, |functions, _| {
        // SAFETY: `functions.hooks` was made by `functions_for` and the
        // capsule (the only owner) is being destroyed.
        unsafe { free_hooks(functions.hooks) }
    })
}

fn functions_for(hooks: Box<Hooks>) -> NativeHooksFunctions {
    let registered_hooks = Hook::ALL
        .into_iter()
        .filter(|hook| hooks.has(*hook))
        .fold(0, |bits, hook| bits | hook.bit());
    NativeHooksFunctions {
        interface_version: INTERFACE_VERSION,
        // `Box::into_raw` gives up ownership: `free_hooks` takes it back.
        hooks: Box::into_raw(hooks) as *const c_void,
        registered_hooks,
        call: call_native,
        free_result,
    }
}

/// # Safety
/// `hooks` must come from `functions_for` and be freed only once.
unsafe fn free_hooks(hooks: *const c_void) {
    drop(unsafe { Box::from_raw(hooks as *mut Hooks) });
}

unsafe extern "C" fn call_native(
    hooks: *const c_void,
    hook: u32,
    context_pointer: *const u8,
    context_length: usize,
    arguments_pointer: *const u8,
    arguments_length: usize,
) -> NativeResult {
    // SAFETY: the engine passes the hooks pointer from this same capsule
    // (kept alive while in use) and two byte slices it owns during the call.
    let (hooks, context_json, arguments_json) = unsafe {
        (
            &*(hooks as *const Hooks),
            std::slice::from_raw_parts(context_pointer, context_length),
            std::slice::from_raw_parts(arguments_pointer, arguments_length),
        )
    };

    // A panic must never cross into another library, so we catch it here.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let hook = Hook::from_number(hook).ok_or(format!("Unknown hook number {hook}"))?;
        let context: RunContext =
            serde_json::from_slice(context_json).map_err(|error| format!("Invalid hook context: {error}"))?;
        let arguments: Vec<Value> =
            serde_json::from_slice(arguments_json).map_err(|error| format!("Invalid hook arguments: {error}"))?;
        hooks.call(hook, &context, arguments)
    }));
    NativeResult::new(outcome.unwrap_or_else(|_| Err("A Rust hook crashed".to_string())))
}

// ---------------------------------------------------------------------------
// Engine side: read the capsule and call the hooks
// ---------------------------------------------------------------------------

/// A set of Rust hooks the engine can call directly.
///
/// It only holds pointers: whoever uses it must keep the Python hooks object
/// (which owns the capsule) alive.
#[derive(Clone, Copy)]
pub struct NativeHooks {
    functions: NativeHooksFunctions,
}

impl NativeHooks {
    /// Reads the native functions of a Python object, if it has them.
    pub fn from_python(hooks: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let Ok(capsule) = hooks.getattr(CAPSULE_ATTRIBUTE) else {
            return Ok(None);
        };
        let capsule = capsule.cast::<PyCapsule>()?;
        let pointer = capsule.pointer_checked(Some(CAPSULE_NAME))?;
        // SAFETY: the capsule name guarantees it holds a `NativeHooksFunctions`
        // (created by `capsule_for`). We copy it out right away.
        let functions = unsafe { *pointer.cast::<NativeHooksFunctions>().as_ref() };
        if functions.interface_version != INTERFACE_VERSION {
            return Err(PyValueError::new_err(format!(
                "Hooks were built with native interface version {}, but this pyox-agents uses version {}",
                functions.interface_version, INTERFACE_VERSION
            )));
        }
        Ok(Some(NativeHooks { functions }))
    }

    /// Whether this set has `hook`, so the engine can skip it without a call.
    pub fn has(&self, hook: Hook) -> bool {
        self.functions.registered_hooks & hook.bit() != 0
    }

    /// Runs one hook. `Ok(json)` is what the hook may have changed (empty
    /// text if it changes nothing), `Err(message)` the hook's error.
    pub fn call(&self, hook: Hook, context_json: &str, arguments_json: &str) -> Result<String, String> {
        // SAFETY: the functions come from a checked capsule that the caller
        // keeps alive. Both texts stay alive during the call, and the result
        // is freed once, by the library that allocated it.
        unsafe {
            let result = (self.functions.call)(
                self.functions.hooks,
                hook as u32,
                context_json.as_ptr(),
                context_json.len(),
                arguments_json.as_ptr(),
                arguments_json.len(),
            );
            result.read_and_free(self.functions.free_result)
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn hooks_change_arguments_and_errors_and_panics_cross_the_interface() {
        let hooks = Hooks::new()
            .before_tool_call(|context, call| match call.name.as_str() {
                "shell" => Err(format!("No shell in turn {}", context.turn)),
                "bug" => panic!("simulated bug"),
                _ => {
                    call.arguments["checked"] = json!(true);
                    Ok(())
                }
            })
            .after_run(|_, _| Ok(()));
        let functions = functions_for(Box::new(hooks));
        let native_hooks = NativeHooks { functions };
        let context = r#"{"conversation_id":null,"turn":2,"input_tokens":0,"output_tokens":0}"#;

        assert!(native_hooks.has(Hook::BeforeToolCall));
        assert!(native_hooks.has(Hook::AfterRun));
        assert!(!native_hooks.has(Hook::BeforeModelRequest));

        let call = |name: &str| {
            let arguments = json!([{"id": "call_1", "name": name, "arguments": {}}]).to_string();
            native_hooks.call(Hook::BeforeToolCall, context, &arguments)
        };
        assert_eq!(
            call("add"),
            Ok(r#"{"id":"call_1","name":"add","arguments":{"checked":true}}"#.into())
        );
        assert_eq!(call("shell"), Err("No shell in turn 2".into()));
        assert_eq!(call("bug"), Err("A Rust hook crashed".into()));
        assert!(
            native_hooks
                .call(Hook::BeforeToolCall, context, "[]")
                .unwrap_err()
                .contains("missing")
        );
        assert_eq!(native_hooks.call(Hook::AfterRun, context, "[[]]"), Ok(String::new()));

        // SAFETY: made by `functions_for` above, freed once.
        unsafe { free_hooks(functions.hooks) };
    }
}
