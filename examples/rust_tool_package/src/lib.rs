//! Example of tools written in Rust and compiled as their own Python package.
//!
//! Build and install into the current virtual environment with:
//!     uv pip install ./examples/rust_tool_package     (or: maturin develop --release)
//!
//! Then in Python:
//!     import fast_tools
//!     from pyox_agents import Tool
//!     count_words = Tool(name=..., description=..., parameters=..., function=fast_tools.count_words, run_in_thread=True)

use pyo3::prelude::*;

/// Counts the words in a text.
///
/// `py.detach(...)` releases Python's global lock (the GIL) while Rust works,
/// so other Python threads keep running during long computations.
#[pyfunction]
fn count_words(py: Python<'_>, text: String) -> usize {
    py.detach(move || text.split_whitespace().count())
}

/// Returns the `top` most frequent words with their counts.
#[pyfunction]
#[pyo3(signature = (text, top = 5))]
fn most_common_words(py: Python<'_>, text: String, top: usize) -> Vec<(String, usize)> {
    py.detach(move || {
        let mut counts = std::collections::HashMap::<String, usize>::new();
        for word in text.split_whitespace() {
            *counts.entry(word.to_lowercase()).or_default() += 1;
        }
        let mut sorted_counts: Vec<(String, usize)> = counts.into_iter().collect();
        // Most frequent first; ties in alphabetical order so results are stable.
        sorted_counts.sort_by(|first, second| second.1.cmp(&first.1).then(first.0.cmp(&second.0)));
        sorted_counts.truncate(top);
        sorted_counts
    })
}

/// The Python module: `import fast_tools`.
#[pymodule]
fn fast_tools(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(count_words, module)?)?;
    module.add_function(wrap_pyfunction!(most_common_words, module)?)?;
    Ok(())
}
