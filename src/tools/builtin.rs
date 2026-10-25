//! Tools written in Rust and compiled into the library.
//!
//! To add one:
//! 1. Write a function `fn my_tool(arguments: &Map<String, Value>) -> Result<String, String>`.
//!    Return `Ok(text)` for the result or `Err(message)` for an error the model will see.
//! 2. Add a `BuiltinToolSpec` for it to `BUILTIN_TOOLS` below.
//!
//! Python then uses it with `BuiltinTool("my_tool")`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

/// The description of a built-in tool.
pub struct BuiltinToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON schema of the arguments, written as JSON text so it is easy to read.
    pub parameters_json: &'static str,
    /// The Rust function that does the work.
    pub function: fn(&Map<String, Value>) -> Result<String, String>,
}

impl BuiltinToolSpec {
    pub fn parameters(&self) -> Value {
        serde_json::from_str(self.parameters_json).expect("built-in tool schemas are valid JSON")
    }
}

/// All built-in tools.
pub static BUILTIN_TOOLS: &[BuiltinToolSpec] = &[
    BuiltinToolSpec {
        name: "read_file",
        description: "Read a UTF-8 text file and return its contents.",
        parameters_json: r#"{
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path of the file to read"}},
            "required": ["path"]
        }"#,
        function: read_file,
    },
    BuiltinToolSpec {
        name: "write_file",
        description: "Write text to a file, creating parent folders and replacing existing content.",
        parameters_json: r#"{
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to write"},
                "content": {"type": "string", "description": "Text to write"}
            },
            "required": ["path", "content"]
        }"#,
        function: write_file,
    },
];

fn read_file(arguments: &Map<String, Value>) -> Result<String, String> {
    let path = string_argument(arguments, "path")?;
    std::fs::read_to_string(path).map_err(|error| format!("Could not read {path}: {error}"))
}

fn write_file(arguments: &Map<String, Value>) -> Result<String, String> {
    let path = string_argument(arguments, "path")?;
    let content = string_argument(arguments, "content")?;
    if let Some(parent_folder) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent_folder)
            .map_err(|error| format!("Could not create folder for {path}: {error}"))?;
    }
    std::fs::write(path, content).map_err(|error| format!("Could not write {path}: {error}"))?;
    Ok(format!("Wrote {} bytes to {path}", content.len()))
}

/// Gets a required string argument, or an error message for the model.
fn string_argument<'a>(arguments: &'a Map<String, Value>, name: &str) -> Result<&'a str, String> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Missing string argument {name:?}"))
}

/// Python: `BuiltinTool("read_file")`. Selects a built-in Rust tool by name.
#[pyclass(frozen, module = "pyox_agents")]
pub struct BuiltinTool {
    pub spec: &'static BuiltinToolSpec,
}

#[pymethods]
impl BuiltinTool {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        BUILTIN_TOOLS
            .iter()
            .find(|spec| spec.name == name)
            .map(|spec| BuiltinTool { spec })
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Unknown built-in tool {name:?}. Available: {:?}",
                    Self::available()
                ))
            })
    }

    /// Names of all built-in tools.
    #[staticmethod]
    fn available() -> Vec<&'static str> {
        BUILTIN_TOOLS.iter().map(|spec| spec.name).collect()
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.spec.name
    }

    #[getter]
    fn description(&self) -> &'static str {
        self.spec.description
    }

    fn __repr__(&self) -> String {
        format!("BuiltinTool({:?})", self.spec.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_schema_is_valid_json() {
        for spec in BUILTIN_TOOLS {
            assert!(spec.parameters().is_object(), "{}", spec.name);
        }
    }

    #[test]
    fn write_then_read_a_file() {
        let path = std::env::temp_dir().join("pyox_agents_test/hello.txt");
        let path = path.to_str().unwrap();
        let mut arguments = Map::new();
        arguments.insert("path".into(), path.into());
        arguments.insert("content".into(), "hola".into());

        assert!(write_file(&arguments).is_ok());
        assert_eq!(read_file(&arguments).unwrap(), "hola");
        assert!(read_file(&Map::new()).unwrap_err().contains("path"));
    }
}
