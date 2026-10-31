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
//! A tool message can also hold data the model never sees: `metadata` (from
//! `ToolReturn`), the `requests` the call made to a person (approvals,
//! `ctx.ask` questions) with their answers, and saved `steps` results. A
//! call whose last request has no answer yet is *waiting*: the run stopped
//! there, and a later run continues it (see `human_input` in pyox-agents).
//! Waiting messages are never sent to the model.
//!
//! A tool call that ran a subagent keeps the subagent's own conversation in
//! `subagent`. The call waits while that conversation does.
//!
//! Stored history must keep working when this format grows, so new fields
//! are always optional (`#[serde(default)]`).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
        /// Data for the app, never sent to the model (`ToolReturn(metadata=...)`).
        #[serde(default, skip_serializing_if = "Value::is_null")]
        metadata: Value,
        /// What the call asked a person, in order, with the answers so far.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        requests: Vec<Request>,
        /// Saved results of `ctx.step(name, ...)`, by name, reused when the call runs again.
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        steps: Map<String, Value>,
        /// The conversation of the subagent this call ran, if it ran one.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        subagent: Vec<Message>,
    },
}

impl Message {
    /// A tool result with nothing but its text (no metadata, requests or steps).
    pub fn tool_result(tool_call_id: String, name: String, content: String, is_error: bool) -> Self {
        Message::Tool {
            tool_call_id,
            name,
            content,
            is_error,
            metadata: Value::Null,
            requests: Vec::new(),
            steps: Map::new(),
            subagent: Vec::new(),
        }
    }

    /// True for a tool message whose call is still waiting for an answer:
    /// one of its requests has none, or its subagent waits for one.
    pub fn is_waiting(&self) -> bool {
        match self {
            Message::Tool { requests, subagent, .. } => {
                requests.iter().any(|request| request.answer.is_none()) || !waiting_calls(subagent).is_empty()
            }
            _ => false,
        }
    }
}

/// The tool calls of the last assistant message that are still waiting, each
/// with its newest tool message. (A resumed call adds a new tool message for
/// the same call: the history only grows.)
pub fn waiting_calls(messages: &[Message]) -> Vec<(ToolCall, &Message)> {
    let Some(assistant_position) = messages
        .iter()
        .rposition(|message| matches!(message, Message::Assistant { .. }))
    else {
        return Vec::new();
    };
    let Message::Assistant { tool_calls, .. } = &messages[assistant_position] else {
        unreachable!("rposition found an assistant message");
    };
    let later_messages = &messages[assistant_position + 1..];
    tool_calls
        .iter()
        .filter_map(|call| {
            let newest = later_messages
                .iter()
                .rev()
                .find(|message| matches!(message, Message::Tool { tool_call_id, .. } if *tool_call_id == call.id))?;
            newest.is_waiting().then(|| (call.clone(), newest))
        })
        .collect()
}

/// Something a tool call asked a person (or the app), and the answer once it came.
///
/// ```python
/// {"kind": "input", "data": {"question": "Which flight?"}, "metadata": {"widget": "picker"},
///  "answer": {"type": "answer", "value": "F123", "metadata": {"card": "tok_9"}}}
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub kind: RequestKind,
    /// The name given with `ctx.ask(..., key=...)`. Answers are matched by key
    /// when there is one, otherwise by position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Shown to the person. Reaches the model only if the tool puts it in its result.
    #[serde(default)]
    pub data: Value,
    /// For the app only (for example, how to display the question). Never sent to the model.
    #[serde(default)]
    pub metadata: Value,
    /// `None` until the person answers.
    #[serde(default)]
    pub answer: Option<Reply>,
}

/// Why a tool call is asking.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    /// May the tool run? Answered with `Approve` or `Deny`.
    Approval,
    /// A question from `ctx.ask(...)`. Answered with `Answer` or `Deny`.
    Input,
    /// A frontend tool: the app runs it and answers with its result.
    Frontend,
}

impl RequestKind {
    /// The name used in Python and in the history: `"approval"`, `"input"` or `"frontend"`.
    pub fn name(self) -> &'static str {
        match self {
            RequestKind::Approval => "approval",
            RequestKind::Input => "input",
            RequestKind::Frontend => "frontend",
        }
    }
}

/// The person's reply to a request. In Python: `Answer`, `Approve` or `Deny`.
/// `#[serde(tag = "type")]` stores the variant name in a `"type"` field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Reply {
    Answer {
        #[serde(default)]
        value: Value,
        #[serde(default)]
        metadata: Value,
    },
    Approve {
        /// Replaces the arguments the model chose.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        arguments: Option<Value>,
        #[serde(default)]
        metadata: Value,
    },
    Deny {
        /// What the model reads as the tool's (error) result.
        message: String,
        #[serde(default)]
        metadata: Value,
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
    fn tool_messages_keep_requests_and_steps_and_know_when_they_wait() {
        // Stored before `metadata`, `requests` and `steps` existed.
        let old: Message = serde_json::from_value(json!({
            "role": "tool", "tool_call_id": "call_1", "name": "add", "content": "3"
        }))
        .unwrap();
        assert_eq!(
            old,
            Message::tool_result("call_1".into(), "add".into(), "3".into(), false)
        );
        assert!(!old.is_waiting());

        let waiting_dict = json!({
            "role": "tool", "tool_call_id": "call_2", "name": "book", "content": "", "is_error": false,
            "requests": [
                {"kind": "approval", "data": null, "metadata": null,
                 "answer": {"type": "approve", "metadata": {"by": "ana"}}},
                {"kind": "input", "key": "flight", "data": {"question": "Which?"}, "metadata": null, "answer": null},
            ],
            "steps": {"search": ["F1", "F2"]},
        });
        let waiting: Message = serde_json::from_value(waiting_dict.clone()).unwrap();
        assert!(waiting.is_waiting());
        // Written back exactly as it was read.
        assert_eq!(serde_json::to_value(&waiting).unwrap(), waiting_dict);
    }

    #[test]
    fn a_tool_message_waits_while_its_subagent_waits() {
        let subagent_waiting: Message = serde_json::from_value(json!({
            "role": "tool", "tool_call_id": "call_0", "name": "research", "content": "",
            "subagent": [
                {"role": "user", "content": "Find flights"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "name": "book", "arguments": {}}]},
                {"role": "tool", "tool_call_id": "call_0", "name": "book", "content": "",
                 "requests": [{"kind": "approval", "answer": null}]},
            ],
        }))
        .unwrap();
        assert!(subagent_waiting.is_waiting());

        let Message::Tool { mut subagent, .. } = subagent_waiting else {
            unreachable!()
        };
        subagent.push(Message::tool_result(
            "call_0".into(),
            "book".into(),
            "Booked".into(),
            false,
        ));
        assert!(waiting_calls(&subagent).is_empty());
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
