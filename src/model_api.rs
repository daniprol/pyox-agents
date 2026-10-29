//! Which API the agent talks to, and the one call the agent loop makes per turn.
//!
//! Each API has its own adapter file: `openai_chat.rs` and `openai_responses.rs`.
//! Both read a `ModelRequest` and give back an `AssistantTurn`, so the agent
//! loop never needs to know which API it uses.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::client::ClientSettings;
use crate::messages::{Message, Reasoning, TokenUsage, ToolCall};
use crate::text_chunks::TextOutput;
use crate::tools::ToolDefinition;
use crate::{openai_chat, openai_responses};

/// Python: `Agent(api="chat")` or `Agent(api="responses")`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Api {
    /// `POST /chat/completions`: OpenAI and most compatible servers.
    ChatCompletions,
    /// `POST /responses`: OpenAI's newer API, with stored responses and encrypted reasoning.
    Responses(ResponsesOptions),
}

/// Settings that only exist in the Responses API.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResponsesOptions {
    /// Let OpenAI keep each response, so the next request can continue from it
    /// (`previous_response_id`) instead of sending the whole history again.
    pub store: bool,
    /// Ask for the model's reasoning in encrypted form, to send it back on
    /// later requests. Models that do not reason (e.g. gpt-4.1) reject this.
    pub encrypted_reasoning: bool,
}

impl Api {
    pub fn from_python(api: &str, store: bool, encrypted_reasoning: bool) -> PyResult<Self> {
        match api {
            "chat" if store => Err(PyValueError::new_err("`store=True` needs `api=\"responses\"`")),
            "chat" => Ok(Api::ChatCompletions),
            "responses" => Ok(Api::Responses(ResponsesOptions {
                store,
                encrypted_reasoning,
            })),
            other => Err(PyValueError::new_err(format!(
                "Unknown api {other:?}: use \"chat\" or \"responses\""
            ))),
        }
    }

    /// True when OpenAI keeps the responses, so requests can use `previous_response_id`.
    pub fn stores_responses(&self) -> bool {
        matches!(self, Api::Responses(options) if options.store)
    }
}

/// Everything one model request needs.
///
/// `<'a>` (a "lifetime") says the fields are borrowed, like references in
/// Python: the request only looks at data the agent and the run own, it does
/// not copy it.
pub struct ModelRequest<'a> {
    pub model: &'a str,
    pub system_prompt: &'a str,
    /// The whole conversation so far. An adapter may send only part of it
    /// (see `previous_response_id`).
    pub messages: &'a [Message],
    pub tools: &'a [ToolDefinition],
    /// Extra fields for the request body (temperature, reasoning, ...).
    pub model_parameters: &'a Map<String, Value>,
    /// Responses API with `store`: the stored response this request continues
    /// from. Only the messages after it are sent. The Chat API ignores it.
    pub previous_response_id: Option<&'a str>,
}

/// What the model answered in one turn.
#[derive(Debug, Default, PartialEq)]
pub struct AssistantTurn {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    /// Only the Responses API returns reasoning and a response id.
    pub reasoning: Vec<Reasoning>,
    pub response_id: Option<String>,
    pub usage: TokenUsage,
}

/// Sends one request to the agent's API and streams the answer: each piece of
/// text goes to `text_output` as soon as it arrives.
pub async fn stream_model_turn(
    client: &ClientSettings,
    api: Api,
    request: &ModelRequest<'_>,
    text_output: &mut TextOutput,
) -> PyResult<AssistantTurn> {
    match api {
        Api::ChatCompletions => openai_chat::stream_chat_completion(client, request, text_output).await,
        Api::Responses(options) => openai_responses::stream_response(client, request, options, text_output).await,
    }
}

/// The id of the newest response in `messages`, if any: the point a stored
/// conversation can continue from.
pub fn last_response_id(messages: &[Message]) -> Option<&str> {
    messages.iter().rev().find_map(|message| match message {
        Message::Assistant {
            response_id: Some(response_id),
            ..
        } => Some(response_id.as_str()),
        _ => None,
    })
}

/// Small helpers shared by the adapters' tests.
#[cfg(test)]
pub mod test_helpers {
    use super::*;

    /// Messages written as the dicts Python uses: `messages(json!([{"role": "user", ...}]))`.
    pub fn messages(dicts: Value) -> Vec<Message> {
        serde_json::from_value(dicts).expect("test messages must be valid")
    }

    /// A request for model "m" with the given messages and nothing else.
    pub fn request<'a>(messages: &'a [Message], model_parameters: &'a Map<String, Value>) -> ModelRequest<'a> {
        ModelRequest {
            model: "m",
            system_prompt: "",
            messages,
            tools: &[],
            model_parameters,
            previous_response_id: None,
        }
    }
}
