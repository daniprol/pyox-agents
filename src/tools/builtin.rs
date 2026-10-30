//! Tools written in Rust and compiled into the library.
//!
//! Each tool is a struct holding its arguments plus a `run` method (the
//! `RustTool` trait from the `pyox-tools` crate). The JSON schema the model
//! sees is generated from the struct; doc comments become descriptions.
//!
//! To add one: write the struct, implement `RustTool`, and add it to
//! `BUILTIN_TOOLS`. Python then uses it with `BuiltinTool("its_name")`.

use std::sync::LazyLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBool;
use pyox_tools::{RustTool, ToolDefinition};
use schemars::JsonSchema;
use serde::Deserialize;

/// All built-in tools. Built once, on first use.
pub static BUILTIN_TOOLS: LazyLock<Vec<ToolDefinition>> =
    LazyLock::new(|| vec![ToolDefinition::of::<ReadFile>(), ToolDefinition::of::<WriteFile>()]);

/// Read a UTF-8 text file and return its contents.
#[derive(Deserialize, JsonSchema)]
struct ReadFile {
    /// Path of the file to read.
    path: String,
}

impl RustTool for ReadFile {
    const NAME: &'static str = "read_file";

    fn run(self) -> Result<String, String> {
        std::fs::read_to_string(&self.path).map_err(|error| format!("Could not read {}: {error}", self.path))
    }
}

/// Write text to a file, creating parent folders and replacing existing content.
#[derive(Deserialize, JsonSchema)]
struct WriteFile {
    /// Path of the file to write.
    path: String,
    /// Text to write.
    content: String,
}

impl RustTool for WriteFile {
    const NAME: &'static str = "write_file";

    fn run(self) -> Result<String, String> {
        let path = &self.path;
        if let Some(parent_folder) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent_folder)
                .map_err(|error| format!("Could not create folder for {path}: {error}"))?;
        }
        std::fs::write(path, &self.content).map_err(|error| format!("Could not write {path}: {error}"))?;
        Ok(format!("Wrote {} bytes to {path}", self.content.len()))
    }
}

/// Python: `BuiltinTool("read_file")`. Selects a built-in Rust tool by name.
/// `requires_approval` works as for Python tools: `True`, or a function
/// `(ctx, arguments) -> bool` deciding for each call.
#[pyclass(frozen, module = "pyox_agents")]
pub struct BuiltinTool {
    pub definition: &'static ToolDefinition,
    #[pyo3(get)]
    requires_approval: Py<PyAny>,
}

#[pymethods]
impl BuiltinTool {
    #[new]
    #[pyo3(signature = (name, *, requires_approval = None))]
    fn new(py: Python<'_>, name: &str, requires_approval: Option<Py<PyAny>>) -> PyResult<Self> {
        let definition = BUILTIN_TOOLS
            .iter()
            .find(|definition| definition.name == name)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Unknown built-in tool {name:?}. Available: {:?}",
                    Self::available()
                ))
            })?;
        Ok(BuiltinTool {
            definition,
            requires_approval: requires_approval
                .unwrap_or_else(|| PyBool::new(py, false).to_owned().into_any().unbind()),
        })
    }

    /// Names of all built-in tools.
    #[staticmethod]
    fn available() -> Vec<&'static str> {
        BUILTIN_TOOLS.iter().map(|definition| definition.name).collect()
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.definition.name
    }

    #[getter]
    fn description(&self) -> &'static str {
        &self.definition.description
    }

    fn __repr__(&self) -> String {
        format!("BuiltinTool({:?})", self.definition.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn schemas_come_from_the_structs() {
        let write_file = BUILTIN_TOOLS.iter().find(|tool| tool.name == "write_file").unwrap();
        assert_eq!(write_file.parameters["required"], json!(["path", "content"]));
        assert_eq!(
            write_file.parameters["properties"]["content"]["description"],
            json!("Text to write.")
        );
    }

    #[test]
    fn write_then_read_a_file() {
        let path = std::env::temp_dir().join("pyox_agents_test/hello.txt");
        let path = path.to_str().unwrap().to_string();

        let written = WriteFile {
            path: path.clone(),
            content: "hola".into(),
        }
        .run();
        assert!(written.is_ok());
        assert_eq!(ReadFile { path }.run().unwrap(), "hola");
    }
}
