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
//! {"role": "assistant", "content": "3", "model": "gpt-5-mini", "response_id": "resp_1",
//!  "reasoning": [{"id": "rs_1", "summary": ["Adding..."], "encrypted_content": "gAAAA..."}]}
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
        /// The model's reasoning (Responses API only), kept to send back
        /// unchanged on the next request. `skip_serializing_if` leaves the
        /// field out of the dict when it is empty.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        reasoning: Vec<Reasoning>,
        /// The id the Responses API gave this answer. With `store=True` the
        /// next request can send it as `previous_response_id` instead of the
        /// whole history.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
        /// The model that wrote this message. Reasoning is only sent back to
        /// this same model: other models cannot read it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
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

/// One reasoning step of the model, as the Responses API returns it.
///
/// The readable part is only a summary. The full reasoning comes encrypted:
/// we cannot read it, but sending it back lets the model continue from it
/// without the server storing anything (`store=False`). Sending it back
/// unchanged also keeps the start of the prompt identical, so OpenAI's prompt
/// cache keeps working.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reasoning {
    pub id: String,
    /// The summary parts, kept apart (not joined) so they go back exactly as they came.
    #[serde(default)]
    pub summary: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn old_assistant_messages_load_and_new_fields_are_left_out_when_empty() {
        // Stored before `reasoning`, `response_id` and `model` existed.
        let message: Message = serde_json::from_value(json!({"role": "assistant", "content": "Hi"})).unwrap();
        let Message::Assistant {
            reasoning,
            response_id,
            model,
            ..
        } = &message
        else {
            panic!("expected an assistant message, got {message:?}");
        };
        assert!(reasoning.is_empty() && response_id.is_none() && model.is_none());
        assert_eq!(
            serde_json::to_value(&message).unwrap(),
            json!({"role": "assistant", "content": "Hi", "tool_calls": []})
        );
    }

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
