//! Example: agent tools written in Rust, compiled as their own Python package.
//!
//! Each tool is a struct with its arguments and a `run` method. The JSON
//! schema the model sees is generated from the struct, and doc comments
//! become the descriptions, so nothing has to be repeated in Python.

use pyo3::prelude::*;
use pyox_tools::{RustTool, add_tool};
use schemars::JsonSchema;
use serde::Deserialize;

/// Count the words in a text.
#[derive(Deserialize, JsonSchema)]
struct CountWords {
    /// The text to count words in.
    text: String,
}

impl RustTool for CountWords {
    const NAME: &'static str = "count_words";

    fn run(self) -> Result<String, String> {
        Ok(self.text.split_whitespace().count().to_string())
    }
}

/// Find the most frequent words in a text, with how many times each appears.
#[derive(Deserialize, JsonSchema)]
struct MostCommonWords {
    /// The text to analyze.
    text: String,
    /// How many words to return (default 5).
    top: Option<usize>,
}

impl RustTool for MostCommonWords {
    const NAME: &'static str = "most_common_words";

    fn run(self) -> Result<String, String> {
        let mut counts = std::collections::HashMap::<String, usize>::new();
        for word in self.text.split_whitespace() {
            *counts.entry(word.to_lowercase()).or_default() += 1;
        }
        let mut sorted_counts: Vec<(String, usize)> = counts.into_iter().collect();
        // Most frequent first; ties in alphabetical order so results are stable.
        sorted_counts.sort_by(|first, second| second.1.cmp(&first.1).then(first.0.cmp(&second.0)));
        sorted_counts.truncate(self.top.unwrap_or(5));

        let lines: Vec<String> = sorted_counts
            .iter()
            .map(|(word, count)| format!("{word}: {count}"))
            .collect();
        Ok(lines.join("\n"))
    }
}

/// The Python module: `import fast_tools` gives `fast_tools.count_words`
/// and `fast_tools.most_common_words`, ready for `Agent(tools=[...])`.
#[pymodule]
fn fast_tools(module: &Bound<'_, PyModule>) -> PyResult<()> {
    add_tool::<CountWords>(module)?;
    add_tool::<MostCommonWords>(module)?;
    Ok(())
}
