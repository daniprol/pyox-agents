//! Example: agent tools and hooks written in Rust, compiled as their own
//! Python package.
//!
//! Each tool is a struct with its arguments and a `run` method. The JSON
//! schema the model sees is generated from the struct, and doc comments
//! become the descriptions, so nothing has to be repeated in Python.
//!
//! Hooks are closures the agent calls at fixed points of a run (see `guard`).

use pyo3::prelude::*;
use pyox_tools::hooks::{Hooks, add_hooks};
use pyox_tools::messages::Message;
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

/// Tool calls with a longer `text` argument are blocked.
const MAX_TEXT_LENGTH: usize = 10_000;

/// Each model request gets at most this many messages.
const MAX_MESSAGES_PER_REQUEST: usize = 20;

/// Hooks for `Agent(hooks=[fast_tools.guard])`:
///
/// - Blocks tool calls whose `text` is too long; the model is told why.
/// - Sends only the recent messages to the model, to keep requests small
///   (the history itself keeps everything).
fn guard() -> Hooks {
    Hooks::new()
        .before_tool_call(|_context, call| {
            let text_length = call.arguments["text"].as_str().map_or(0, str::len);
            if text_length > MAX_TEXT_LENGTH {
                return Err(format!(
                    "The text is too long ({text_length} bytes, at most {MAX_TEXT_LENGTH})"
                ));
            }
            Ok(())
        })
        .before_model_request(|_context, messages| {
            let first_kept = messages.len().saturating_sub(MAX_MESSAGES_PER_REQUEST);
            messages.drain(..first_kept);
            // Start at a user message: tool results without the assistant
            // message that asked for them would confuse the API.
            let first_user_message = messages
                .iter()
                .position(|message| matches!(message, Message::User { .. }))
                .unwrap_or(0);
            messages.drain(..first_user_message);
            Ok(())
        })
}

/// The Python module: `import fast_tools` gives `fast_tools.count_words`
/// and `fast_tools.most_common_words`, ready for `Agent(tools=[...])`, and
/// `fast_tools.guard`, ready for `Agent(hooks=[...])`.
#[pymodule]
fn fast_tools(module: &Bound<'_, PyModule>) -> PyResult<()> {
    add_tool::<CountWords>(module)?;
    add_tool::<MostCommonWords>(module)?;
    add_hooks(module, "guard", guard())?;
    Ok(())
}
