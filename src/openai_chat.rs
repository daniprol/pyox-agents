//! Adapter for the OpenAI Chat Completions API (`POST /chat/completions`).
//!
//! It translates in both directions:
//! - our `Message` history  ->  the JSON request body the API expects
//! - the streamed response  ->  `TextDelta` events + one finished `AssistantTurn`
//!
//! Supporting another API (Responses, Anthropic, ...) means writing another
//! file like this one; the agent loop does not change.

use futures::StreamExt;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::client::ClientSettings;
use crate::errors::{ApiError, InvalidResponseError, error_for_http_failure, error_for_http_status};
use crate::messages::{Message, TokenUsage, ToolCall};
use crate::sse::SseDecoder;
use crate::text_chunks::TextOutput;
use crate::tools::ToolDefinition;

/// What the model answered in one turn.
#[derive(Debug, Default, PartialEq)]
pub struct AssistantTurn {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
}

// ---------------------------------------------------------------------------
// Request: our history -> Chat Completions JSON
// ---------------------------------------------------------------------------

/// Builds the request body. `model_parameters` (temperature, ...) go first so
/// the fields this library controls (`model`, `messages`, `stream`, ...) always win.
pub fn build_request_body(
    model: &str,
    system_prompt: &str,
    history: &[Message],
    tools: &[ToolDefinition],
    model_parameters: &Map<String, Value>,
) -> Value {
    let mut body = model_parameters.clone();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), Value::Array(chat_messages(system_prompt, history)));
    body.insert("stream".into(), json!(true));
    body.insert("stream_options".into(), json!({"include_usage": true}));

    if !tools.is_empty() {
        let tool_schemas = tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            })
            .collect();
        body.insert("tools".into(), Value::Array(tool_schemas));
    }
    Value::Object(body)
}

fn chat_messages(system_prompt: &str, history: &[Message]) -> Vec<Value> {
    let mut chat_messages = Vec::with_capacity(history.len() + 1);
    if !system_prompt.is_empty() {
        chat_messages.push(json!({"role": "system", "content": system_prompt}));
    }

    for message in history {
        let chat_message = match message {
            Message::User { content } => json!({"role": "user", "content": content}),
            Message::Assistant { content, tool_calls } if tool_calls.is_empty() => {
                json!({"role": "assistant", "content": content})
            }
            Message::Assistant { content, tool_calls } => json!({
                "role": "assistant",
                "content": if content.is_empty() { Value::Null } else { json!(content) },
                "tool_calls": tool_calls.iter().map(|call| json!({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments_as_text(&call.arguments)},
                })).collect::<Vec<_>>(),
            }),
            Message::Tool {
                tool_call_id, content, ..
            } => {
                json!({"role": "tool", "tool_call_id": tool_call_id, "content": content})
            }
        };
        chat_messages.push(chat_message);
    }
    chat_messages
}

/// The API wants tool arguments as JSON text. Invalid arguments were stored
/// as a plain string (see `ToolCall`), so we send that string back unchanged.
fn arguments_as_text(arguments: &Value) -> String {
    match arguments {
        Value::String(raw_text) => raw_text.clone(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Response: streamed chunks -> AssistantTurn
// ---------------------------------------------------------------------------

// The shape of one streamed chunk. `Option` fields may be missing or `null`;
// fields we do not need are ignored by serde.

#[derive(Deserialize)]
struct ChatCompletionChunk {
    choices: Option<Vec<ChunkChoice>>,
    usage: Option<ChunkUsage>,
    error: Option<Value>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    delta: Option<ChunkDelta>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChunkDelta {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Tool calls arrive in pieces: the first piece has the id and name, the
/// following ones add fragments of the arguments text. `index` says which
/// tool call a piece belongs to.
#[derive(Deserialize)]
struct ToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct ChunkUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// A tool call being assembled from pieces.
#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments_text: String,
}

/// Collects the chunks of one response into an `AssistantTurn`.
#[derive(Default)]
pub struct ChunkAccumulator {
    text: String,
    tool_calls: Vec<PartialToolCall>,
    usage: TokenUsage,
    /// True once the API said why it stopped; used to detect cut-off streams.
    finished: bool,
}

impl ChunkAccumulator {
    /// Adds one chunk (the JSON text of an SSE event). Returns the new text, if any.
    pub fn add_chunk(&mut self, chunk_json: &str) -> PyResult<Option<String>> {
        let chunk: ChatCompletionChunk = serde_json::from_str(chunk_json)
            .map_err(|error| InvalidResponseError::new_err(format!("{error}: {chunk_json}")))?;

        if let Some(error) = chunk.error {
            return Err(ApiError::new_err(error.to_string()));
        }
        if let Some(usage) = chunk.usage {
            self.usage = TokenUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            };
        }

        let Some(choice) = chunk.choices.and_then(|choices| choices.into_iter().next()) else {
            return Ok(None); // e.g. the final usage-only chunk
        };
        if choice.finish_reason.is_some() {
            self.finished = true;
        }
        let Some(delta) = choice.delta else {
            return Ok(None);
        };

        for tool_call_delta in delta.tool_calls.unwrap_or_default() {
            self.add_tool_call_delta(tool_call_delta);
        }

        let new_text = delta.content.filter(|text| !text.is_empty());
        if let Some(text) = &new_text {
            self.text.push_str(text);
        }
        Ok(new_text)
    }

    fn add_tool_call_delta(&mut self, delta: ToolCallDelta) {
        if self.tool_calls.len() <= delta.index {
            self.tool_calls.resize_with(delta.index + 1, PartialToolCall::default);
        }
        let tool_call = &mut self.tool_calls[delta.index];
        if let Some(id) = delta.id {
            tool_call.id = id;
        }
        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                tool_call.name.push_str(&name);
            }
            if let Some(arguments_fragment) = function.arguments {
                tool_call.arguments_text.push_str(&arguments_fragment);
            }
        }
    }

    /// Finishes the turn, parsing each tool call's complete arguments text.
    pub fn into_turn(self) -> AssistantTurn {
        let tool_calls = self
            .tool_calls
            .into_iter()
            .filter(|tool_call| !tool_call.name.is_empty())
            .map(|tool_call| {
                let arguments = if tool_call.arguments_text.trim().is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(&tool_call.arguments_text).unwrap_or(Value::String(tool_call.arguments_text))
                };
                ToolCall {
                    id: tool_call.id,
                    name: tool_call.name,
                    arguments,
                }
            })
            .collect();
        AssistantTurn {
            text: self.text,
            tool_calls,
            usage: self.usage,
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

/// Sends one request and streams the answer. Each piece of text is sent to
/// `text_output` as soon as it arrives.
pub async fn stream_chat_completion(
    client: &ClientSettings,
    request_body: &Value,
    text_output: &mut TextOutput,
) -> PyResult<AssistantTurn> {
    let response = client
        .post_json("/chat/completions", request_body)
        .await
        .map_err(error_for_http_failure)?;
    let status = response.status();
    if !status.is_success() {
        let response_body = response.text().await.unwrap_or_default();
        return Err(error_for_http_status(status.as_u16(), &response_body));
    }

    let mut sse_decoder = SseDecoder::default();
    let mut accumulator = ChunkAccumulator::default();
    let mut received_done = false;
    let mut byte_stream = response.bytes_stream();

    'reading: while let Some(network_chunk) = byte_stream.next().await {
        let network_chunk = network_chunk.map_err(error_for_http_failure)?;
        for event_data in sse_decoder.push(&network_chunk) {
            if event_data == "[DONE]" {
                received_done = true;
                break 'reading;
            }
            if let Some(text) = accumulator.add_chunk(&event_data)? {
                text_output.push(&text).await;
            }
        }
    }
    if !received_done
        && let Some(event_data) = sse_decoder.finish()
        && event_data != "[DONE]"
    {
        accumulator.add_chunk(&event_data)?;
    }

    if !received_done && !accumulator.finished {
        return Err(InvalidResponseError::new_err("The response stream ended unexpectedly"));
    }
    Ok(accumulator.into_turn())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_and_tool_call_fragments_are_assembled() {
        let mut accumulator = ChunkAccumulator::default();
        let chunks = [
            r#"{"choices":[{"delta":{"role":"assistant","content":"Hi"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"add","arguments":"{\"a\":"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#,
        ];
        let texts: Vec<_> = chunks
            .iter()
            .map(|chunk| accumulator.add_chunk(chunk).unwrap())
            .collect();
        assert_eq!(texts[0].as_deref(), Some("Hi"));
        assert!(accumulator.finished);

        let turn = accumulator.into_turn();
        assert_eq!(turn.text, "Hi");
        assert_eq!(
            turn.tool_calls,
            vec![ToolCall {
                id: "call_1".into(),
                name: "add".into(),
                arguments: json!({"a": 1})
            }]
        );
        assert_eq!(
            turn.usage,
            TokenUsage {
                input_tokens: 10,
                output_tokens: 5
            }
        );
    }

    #[test]
    fn invalid_arguments_are_kept_as_text() {
        let mut accumulator = ChunkAccumulator::default();
        accumulator
            .add_chunk(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{oops"}}]}}]}"#)
            .unwrap();
        assert_eq!(accumulator.into_turn().tool_calls[0].arguments, json!("{oops"));
    }

    #[test]
    fn request_body_contains_history_and_tools() {
        let history = vec![
            Message::User {
                content: "add 1".into(),
            },
            Message::Assistant {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "c".into(),
                    name: "add".into(),
                    arguments: json!({"a": 1}),
                }],
            },
            Message::Tool {
                tool_call_id: "c".into(),
                name: "add".into(),
                content: "1".into(),
                is_error: false,
            },
        ];
        let mut parameters = Map::new();
        parameters.insert("temperature".into(), json!(0.2));
        parameters.insert("stream".into(), json!(false)); // must be overridden

        let body = build_request_body("m", "Be brief", &history, &[], &parameters);
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["temperature"], json!(0.2));
        assert_eq!(body["messages"][0], json!({"role": "system", "content": "Be brief"}));
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["arguments"],
            json!("{\"a\":1}")
        );
        assert_eq!(
            body["messages"][3],
            json!({"role": "tool", "tool_call_id": "c", "content": "1"})
        );
        assert!(body.get("tools").is_none());
    }
}
