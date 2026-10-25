//! Calling a tool's Rust function directly, without going through Python.
//!
//! A tool package (compiled with `pyox-tools`) and pyox-agents are separate
//! compiled libraries. Normal Rust types and functions can't safely cross
//! between separately compiled libraries, because Rust doesn't promise a
//! stable memory layout. The C language's conventions *are* stable, so the
//! two libraries talk through a tiny C-style interface:
//!
//! - `NativeToolFunctions`: two function pointers, `run` and `free_result`,
//!   plus a version number. Each tool package creates one per tool and
//!   stores it in a Python "capsule" (a Python object that just holds a pointer).
//! - The engine reads the capsule once, when the `Agent` is created, then
//!   calls `run` directly on every tool call: JSON bytes in, text out.
//!
//! Text returned by `run` was allocated by the tool package, so it must be
//! freed by the tool package too: that's what `free_result` is for.
//!
//! `unsafe` marks code where Rust can't check memory safety by itself; the
//! comments next to each `unsafe` block explain why it is correct.

use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::panic::{AssertUnwindSafe, catch_unwind};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use serde_json::Value;

use crate::{RustTool, parse_and_run};

/// Bump when `NativeToolFunctions` or `NativeToolResult` change.
pub const INTERFACE_VERSION: u32 = 1;

/// The capsule name; checked when reading, so a wrong capsule is rejected.
pub const CAPSULE_NAME: &CStr = c"pyox_tools.native_tool.v1";

/// The Python attribute that holds the capsule on a tool object.
pub const CAPSULE_ATTRIBUTE: &str = "__pyox_native_tool__";

/// Text returned by a native tool, as raw parts of a Rust `Vec<u8>`.
/// `#[repr(C)]` gives the struct the same memory layout in every library.
#[repr(C)]
pub struct NativeToolResult {
    pub is_error: bool,
    pub text_pointer: *mut u8,
    pub text_length: usize,
    pub text_capacity: usize,
}

/// The functions of one native tool.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NativeToolFunctions {
    pub interface_version: u32,
    /// Runs the tool. `arguments` points to the JSON text of the arguments.
    pub run: unsafe extern "C" fn(arguments_pointer: *const u8, arguments_length: usize) -> NativeToolResult,
    /// Frees a result returned by `run`.
    pub free_result: unsafe extern "C" fn(result: NativeToolResult),
}

// ---------------------------------------------------------------------------
// Tool package side: export a `RustTool` as native functions
// ---------------------------------------------------------------------------

/// Makes the capsule for a tool. `add_tool` attaches it to the Python tool object.
pub fn capsule_for<Tool: RustTool + 'static>(py: Python<'_>) -> PyResult<Bound<'_, PyCapsule>> {
    let functions = NativeToolFunctions {
        interface_version: INTERFACE_VERSION,
        run: run_native::<Tool>,
        free_result,
    };
    PyCapsule::new_with_value(py, functions, CAPSULE_NAME)
}

unsafe extern "C" fn run_native<Tool: RustTool>(
    arguments_pointer: *const u8,
    arguments_length: usize,
) -> NativeToolResult {
    // SAFETY: the engine passes a pointer and length of a byte slice it owns
    // and keeps alive until this call returns (see `NativeTool::call`).
    let arguments_bytes = unsafe { std::slice::from_raw_parts(arguments_pointer, arguments_length) };

    // A panic must never cross into another library, so we catch it here.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let arguments: Value =
            serde_json::from_slice(arguments_bytes).map_err(|error| format!("Invalid arguments: {error}"))?;
        parse_and_run::<Tool>(arguments)
    }));
    let (is_error, text) = match outcome {
        Ok(Ok(text)) => (false, text),
        Ok(Err(message)) => (true, message),
        Err(_) => (true, format!("Tool {} crashed", Tool::NAME)),
    };

    // Hand over the text's memory without freeing it; `free_result` frees it later.
    let mut bytes = ManuallyDrop::new(text.into_bytes());
    NativeToolResult {
        is_error,
        text_pointer: bytes.as_mut_ptr(),
        text_length: bytes.len(),
        text_capacity: bytes.capacity(),
    }
}

unsafe extern "C" fn free_result(result: NativeToolResult) {
    // SAFETY: the parts come from a `Vec<u8>` built in `run_native`, in this
    // same library, and each result is freed exactly once.
    drop(unsafe { Vec::from_raw_parts(result.text_pointer, result.text_length, result.text_capacity) });
}

// ---------------------------------------------------------------------------
// Engine side: read the capsule and call the tool
// ---------------------------------------------------------------------------

/// A native tool the engine can call directly.
#[derive(Clone, Copy)]
pub struct NativeTool {
    functions: NativeToolFunctions,
}

impl NativeTool {
    /// Reads the native functions of a Python tool object, if it has them.
    pub fn from_python(tool: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let Ok(capsule) = tool.getattr(CAPSULE_ATTRIBUTE) else {
            return Ok(None);
        };
        let capsule = capsule.cast::<PyCapsule>()?;
        let pointer = capsule.pointer_checked(Some(CAPSULE_NAME))?;
        // SAFETY: the capsule name guarantees it holds a `NativeToolFunctions`
        // (created by `capsule_for`). We copy it out right away.
        let functions = unsafe { *pointer.cast::<NativeToolFunctions>().as_ref() };
        if functions.interface_version != INTERFACE_VERSION {
            return Err(PyValueError::new_err(format!(
                "Tool was built with native interface version {}, but this pyox-agents uses version {}",
                functions.interface_version, INTERFACE_VERSION
            )));
        }
        Ok(Some(NativeTool { functions }))
    }

    /// Runs the tool. `Ok(text)` is the result, `Err(message)` an error for the model.
    pub fn call(&self, arguments: &Value) -> Result<String, String> {
        let arguments_json = arguments.to_string();
        // SAFETY: `run` and `free_result` come from a checked capsule. The
        // arguments slice stays alive during the call, and the result is
        // copied before being freed once by the library that allocated it.
        unsafe {
            let result = (self.functions.run)(arguments_json.as_ptr(), arguments_json.len());
            let text_bytes = std::slice::from_raw_parts(result.text_pointer, result.text_length);
            let text = String::from_utf8_lossy(text_bytes).into_owned();
            let is_error = result.is_error;
            (self.functions.free_result)(result);
            if is_error { Err(text) } else { Ok(text) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use serde_json::json;

    /// Divide two numbers.
    #[derive(Deserialize, JsonSchema)]
    struct Divide {
        dividend: i64,
        divisor: i64,
    }

    impl RustTool for Divide {
        const NAME: &'static str = "divide";

        fn run(self) -> Result<String, String> {
            if self.divisor == 0 {
                return Err("Cannot divide by zero".into());
            }
            if self.divisor == -1 {
                panic!("simulated bug");
            }
            Ok((self.dividend / self.divisor).to_string())
        }
    }

    fn native_divide() -> NativeTool {
        NativeTool {
            functions: NativeToolFunctions {
                interface_version: INTERFACE_VERSION,
                run: run_native::<Divide>,
                free_result,
            },
        }
    }

    #[test]
    fn results_errors_and_panics_cross_the_interface() {
        let tool = native_divide();
        assert_eq!(tool.call(&json!({"dividend": 9, "divisor": 3})), Ok("3".into()));
        assert_eq!(
            tool.call(&json!({"dividend": 9, "divisor": 0})),
            Err("Cannot divide by zero".into())
        );
        assert_eq!(
            tool.call(&json!({"dividend": 9, "divisor": -1})),
            Err("Tool divide crashed".into())
        );
        assert!(
            tool.call(&json!({"dividend": 9}))
                .unwrap_err()
                .starts_with("Invalid arguments")
        );
    }
}
