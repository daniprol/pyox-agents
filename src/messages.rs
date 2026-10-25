//! Our own conversation history format.
//!
//! `Message` and `ToolCall` are defined in the shared `pyox-tools` crate, so
//! Rust history stores compiled separately use the very same types. See
//! `pyox_tools::messages` for the format and how it looks in Python.

pub use pyox_tools::messages::{Message, ToolCall};

/// Token counts reported by the API.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl TokenUsage {
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}
