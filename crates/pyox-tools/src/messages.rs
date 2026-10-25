//! The conversation history format of pyox-agents.
//!
//! The agent loop only works with these types. Each API adapter (for now only
//! OpenAI Chat Completions) translates them into its own request format, so
//! adding another API later does not change the history format.
//!
//! They live in this crate (not in pyox-agents) so that Rust history stores
//! (see [`crate::history`]) build exactly the same types the engine uses.
//!
//! In Python each message is a plain dict, for example:
//!
//! ```python
//! {"role": "user", "content": "Hi"}
//! {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "add", "arguments": {"a": 1}}]}
//! {"role": "tool", "tool_call_id": "call_1", "name": "add", "content": "3", "is_error": False}
//! ```
//!
//! Stored history must keep working when this format grows, so new fields
//! are always optional (`#[serde(default)]`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One entry of the conversation. `#[serde(tag = "role")]` stores the variant
/// name in a `"role"` field, exactly like the dicts above.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    User {
        content: String,
    },
    Assistant {
        #[serde(default)]
        content: String,
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        name: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

/// A tool call requested by the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Normally a JSON object. If the model sent invalid JSON we keep the raw
    /// text as a JSON string, so the model can see (and fix) its mistake.
    pub arguments: Value,
}
