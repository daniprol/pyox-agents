//! Extend pyox-agents in Rust: tools (this page), history stores
//! ([`history`]) and hooks ([`hooks`]), all called by the agent without
//! going through Python.
//!
//! ## Tools
//!
//! A tool is a struct describing its arguments, plus a `run` method:
//!
//! ```ignore
//! use pyox_tools::RustTool;
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! /// Count the words in a text.            <- becomes the tool description
//! #[derive(Deserialize, JsonSchema)]
//! struct CountWords {
//!     /// The text to count words in.       <- becomes the argument description
//!     text: String,
//! }
//!
//! impl RustTool for CountWords {
//!     const NAME: &'static str = "count_words";
//!
//!     fn run(self) -> Result<String, String> {
//!         Ok(self.text.split_whitespace().count().to_string())
//!     }
//! }
//! ```
//!
//! Your crate needs `serde` and `schemars` as dependencies too (for the derives).
//!
//! The JSON schema the model sees is generated from the struct (field names,
//! types, `Option` = not required, doc comments = descriptions), so the
//! arguments are described once, in Rust.
//!
//! To use it from Python, export it from a PyO3 module with [`add_tool`]:
//! `add_tool::<CountWords>(module)?` makes `your_package.count_words`, which
//! can be passed directly in `Agent(tools=[...])`. The agent then calls the
//! Rust function directly, without going through Python (see [`native`]).

pub mod history;
pub mod hooks;
pub mod messages;
pub mod native;
pub mod native_history;
pub mod native_hooks;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Implement this on a struct holding the tool's arguments.
pub trait RustTool: DeserializeOwned + JsonSchema {
    /// The name the model uses to call the tool.
    const NAME: &'static str;

    /// Does the work. Return `Ok(text)` for the result, or `Err(message)` for
    /// an error the model will see (so it can fix its arguments and retry).
    fn run(self) -> Result<String, String>;
}

/// A tool with its types erased, so tools of different kinds fit in one list.
#[derive(Clone, Debug)]
pub struct ToolDefinition {
    pub name: &'static str,
    pub description: String,
    /// JSON schema of the arguments.
    pub parameters: Value,
    /// Parses the JSON arguments into the tool's struct and runs it.
    pub run: fn(Value) -> Result<String, String>,
}

impl ToolDefinition {
    /// Builds the definition of a `RustTool`, generating its schema.
    pub fn of<Tool: RustTool>() -> Self {
        let mut parameters = schemars::schema_for!(Tool).to_value();
        let fields = parameters.as_object_mut().expect("a struct schema is a JSON object");
        // The struct's doc comment is the tool description; the model doesn't need the rest.
        let description = match fields.remove("description") {
            Some(Value::String(text)) => text,
            _ => String::new(),
        };
        fields.remove("$schema");
        fields.remove("title");

        ToolDefinition {
            name: Tool::NAME,
            description,
            parameters,
            run: parse_and_run::<Tool>,
        }
    }
}

fn parse_and_run<Tool: RustTool>(arguments: Value) -> Result<String, String> {
    let tool: Tool = serde_json::from_value(arguments).map_err(|error| format!("Invalid arguments: {error}"))?;
    tool.run()
}

/// The Python object for a Rust tool. pyox-agents reads its `name`,
/// `description` and `parameters`, then calls the Rust function directly
/// through `__pyox_native_tool__`. `invoke(arguments)` lets Python call it too.
#[pyclass(frozen, name = "RustTool")]
pub struct PythonRustTool {
    definition: ToolDefinition,
    native_tool: Py<pyo3::types::PyCapsule>,
}

#[pymethods]
impl PythonRustTool {
    #[getter]
    fn name(&self) -> &'static str {
        self.definition.name
    }

    #[getter]
    fn description(&self) -> &str {
        &self.definition.description
    }

    /// The JSON schema of the arguments, as a dict.
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("json")?
            .call_method1("loads", (self.definition.parameters.to_string(),))
    }

    /// Runs the tool. `arguments` is the dict chosen by the model.
    fn invoke(&self, py: Python<'_>, arguments: Bound<'_, PyAny>) -> PyResult<String> {
        let arguments_json: String = py.import("json")?.call_method1("dumps", (arguments,))?.extract()?;
        let arguments: Value =
            serde_json::from_str(&arguments_json).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let run = self.definition.run;
        // `detach` releases Python's global lock while Rust works.
        py.detach(move || run(arguments)).map_err(PyRuntimeError::new_err)
    }

    /// The capsule with the tool's native functions (see the `native` module).
    #[getter]
    fn __pyox_native_tool__(&self, py: Python<'_>) -> Py<pyo3::types::PyCapsule> {
        self.native_tool.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!("RustTool({:?})", self.definition.name)
    }
}

/// Adds a tool to a PyO3 module, as an attribute named after the tool.
pub fn add_tool<Tool: RustTool + 'static>(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let tool = PythonRustTool {
        definition: ToolDefinition::of::<Tool>(),
        native_tool: native::capsule_for::<Tool>(py)?.unbind(),
    };
    module.add(Tool::NAME, Py::new(py, tool)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    /// Repeat a word.
    #[derive(Deserialize, JsonSchema)]
    struct Repeat {
        /// The word to repeat.
        word: String,
        /// How many times (default 2).
        times: Option<usize>,
    }

    impl RustTool for Repeat {
        const NAME: &'static str = "repeat";

        fn run(self) -> Result<String, String> {
            Ok(self.word.repeat(self.times.unwrap_or(2)))
        }
    }

    #[test]
    fn schema_and_description_come_from_the_struct() {
        let definition = ToolDefinition::of::<Repeat>();
        assert_eq!(definition.name, "repeat");
        assert_eq!(definition.description, "Repeat a word.");
        assert_eq!(definition.parameters["required"], json!(["word"]));
        assert_eq!(definition.parameters["properties"]["word"]["type"], json!("string"));
        assert_eq!(
            definition.parameters["properties"]["word"]["description"],
            json!("The word to repeat.")
        );
        assert!(definition.parameters.get("title").is_none());
    }

    #[test]
    fn run_parses_arguments() {
        let definition = ToolDefinition::of::<Repeat>();
        assert_eq!(
            (definition.run)(json!({"word": "ha", "times": 3})),
            Ok("hahaha".to_string())
        );
        assert!(
            (definition.run)(json!({"times": 3}))
                .unwrap_err()
                .starts_with("Invalid arguments")
        );
    }
}
