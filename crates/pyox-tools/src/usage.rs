//! Token usage: how many tokens a request or a run used.
//!
//! In Python it is `pyox_agents.Usage`, a small read-only object:
//!
//! ```python
//! result.usage              # Usage(input_tokens=1200, cached_input_tokens=900, output_tokens=40, ...)
//! result.usage.input_tokens # 1200
//! total = Usage() + first.usage + second.usage
//! ```
//!
//! It lives in this crate so Rust hooks see the same type in `RunContext`.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Token counts reported by the API, summed over the model requests of a run.
/// Counts the API did not report are 0.
///
/// `eq` gives Python `==`, `get_all` a read-only property for every field,
/// and `from_py_object` lets Rust functions take a `Usage` from Python (for `+`).
#[pyclass(frozen, eq, get_all, from_py_object, module = "pyox_agents")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    /// The part of `input_tokens` read from the prompt cache (cheaper and faster).
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    /// The part of `output_tokens` the model spent reasoning (reasoning models only).
    pub reasoning_tokens: u64,
    /// Model requests that reported usage. Lower than the number of requests
    /// made means some usage is missing (some OpenAI-compatible servers send none).
    pub requests: u32,
}

impl Usage {
    /// Adds the counts of `other` to these, like `+=` in Python.
    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.requests += other.requests;
    }
}

#[pymethods]
impl Usage {
    /// `Usage()` is all zeros, a starting point for summing runs.
    /// `signature` makes every argument keyword-only with a default of 0.
    #[new]
    #[pyo3(signature = (*, input_tokens=0, cached_input_tokens=0, output_tokens=0, reasoning_tokens=0, requests=0))]
    fn new(
        input_tokens: u64,
        cached_input_tokens: u64,
        output_tokens: u64,
        reasoning_tokens: u64,
        requests: u32,
    ) -> Self {
        Usage {
            input_tokens,
            cached_input_tokens,
            output_tokens,
            reasoning_tokens,
            requests,
        }
    }

    fn __add__(&self, other: Usage) -> Usage {
        let mut total = *self;
        total.add(other);
        total
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Usage(input_tokens={}, cached_input_tokens={}, output_tokens={}, reasoning_tokens={}, requests={})",
            self.input_tokens, self.cached_input_tokens, self.output_tokens, self.reasoning_tokens, self.requests
        )
    }
}
