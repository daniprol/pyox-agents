//! Our own conversation history format.
//!
//! The agent loop only works with these types. Each API adapter (for now only
//! `openai_chat`) translates them into its own request format. This way adding
//! another API later does not change the agent loop.
//!
//! In Python each message is a plain dict, for example:
//!
//! ```python
//! {"role": "user", "content": "Hi"}
//! {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "add", "arguments": {"a": 1}}]}
//! {"role": "tool", "tool_call_id": "call_1", "name": "add", "content": "3", "is_error": False}
//! ```

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
