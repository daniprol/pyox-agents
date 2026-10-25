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
//! {"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image", "url": "https://..."}]}
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
        content: UserContent,
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

/// What a user message contains: plain text, or a list of parts when it
/// has images or files. `#[serde(untagged)]` means there is no tag: a JSON
/// string is read as `Text`, a JSON list as `Parts`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One part of a user message. Its `"type"` field says which kind it is.
///
/// Images and files are given by URL: a web address (`https://...`), or a
/// `data:` URL holding the bytes themselves (`data:application/pdf;base64,...`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        url: String,
    },
    /// A document, such as a PDF. The model sees `filename` too.
    File {
        url: String,
        filename: String,
    },
}

/// Lets plain text become user content with `.into()`.
impl From<String> for UserContent {
    fn from(text: String) -> Self {
        UserContent::Text(text)
    }
}

impl From<&str> for UserContent {
    fn from(text: &str) -> Self {
        UserContent::Text(text.to_string())
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn user_content_is_text_or_a_list_of_parts() {
        let messages: Vec<Message> = serde_json::from_value(json!([
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": [
                {"type": "text", "text": "Compare them"},
                {"type": "image", "url": "https://example.com/cat.png"},
                {"type": "file", "url": "data:application/pdf;base64,JVBERg==", "filename": "a.pdf"},
            ]},
        ]))
        .unwrap();

        assert_eq!(messages[0], Message::User { content: "Hi".into() });
        let Message::User {
            content: UserContent::Parts(parts),
        } = &messages[1]
        else {
            panic!("expected parts, got {:?}", messages[1]);
        };
        assert_eq!(
            parts[2],
            ContentPart::File {
                url: "data:application/pdf;base64,JVBERg==".into(),
                filename: "a.pdf".into()
            }
        );
        // Written back exactly as it was read.
        assert_eq!(
            serde_json::to_value(&messages[1]).unwrap()["content"][1]["type"],
            "image"
        );
    }
}
