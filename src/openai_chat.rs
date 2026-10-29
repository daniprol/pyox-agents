//! Adapter for the OpenAI Chat Completions API (`POST /chat/completions`).
//!
//! It translates in both directions:
//! - our `Message` history  ->  the JSON request body the API expects
//! - the streamed response  ->  `TextDelta` events + one finished `AssistantTurn`
//!
//! `openai_responses.rs` does the same for the Responses API, and
//! `model_api.rs` picks one of them; the agent loop does not change.
//! This API has no reasoning to send back, so the `reasoning` of assistant
//! messages is never sent.

use futures::StreamExt;
use pyo3::prelude::*;
use pyox_tools::usage::Usage;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::client::ClientSettings;
use crate::errors::{ApiError, InvalidResponseError, error_for_http_failure, error_for_http_status};
use crate::messages::{ContentPart, Message, ToolCall, UserContent, arguments_as_text, arguments_from_text};
use crate::model_api::{AssistantTurn, ModelRequest};
use crate::sse::SseDecoder;
use crate::text_chunks::TextOutput;

// ---------------------------------------------------------------------------
// Request: our history -> Chat Completions JSON
// ---------------------------------------------------------------------------

/// Builds the request body. `model_parameters` (temperature, ...) go first so
/// the fields this library controls (`model`, `messages`, `stream`, ...) always win.
pub fn build_request_body(request: &ModelRequest<'_>) -> Value {
    let mut body = request.model_parameters.clone();
    body.insert("model".into(), json!(request.model));
    body.insert(
        "messages".into(),
        Value::Array(chat_messages(request.system_prompt, request.messages)),
    );
    body.insert("stream".into(), json!(true));
    body.insert("stream_options".into(), json!({"include_usage": true}));

    if !request.tools.is_empty() {
        let tool_schemas = request
            .tools
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
            Message::User { content } => json!({"role": "user", "content": user_content(content)}),
            Message::Assistant {
                content, tool_calls, ..
            } if tool_calls.is_empty() => {
                json!({"role": "assistant", "content": content})
            }
            Message::Assistant {
                content, tool_calls, ..
            } => json!({
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

/// Text stays a plain string; parts become the API's list of content parts.
fn user_content(content: &UserContent) -> Value {
    match content {
        UserContent::Text(text) => json!(text),
        UserContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => json!({"type": "text", "text": text}),
                ContentPart::Image { url } => json!({"type": "image_url", "image_url": {"url": url}}),
                // OpenAI wants the file bytes as a `data:` URL here; some compatible
                // servers (e.g. OpenRouter) also accept a web address.
                ContentPart::File { url, filename } => {
                    json!({"type": "file", "file": {"file_data": url, "filename": filename}})
                }
            })
            .collect(),
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

/// `#[serde(default)]`: a missing count is read as 0 instead of failing the run.
#[derive(Default, Deserialize)]
#[serde(default)]
struct ChunkUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    prompt_tokens_details: Option<TokensDetails>,
    completion_tokens_details: Option<TokensDetails>,
}

#[derive(Deserialize)]
struct TokensDetails {
    cached_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
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
    usage: Usage,
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
            self.usage = Usage {
                input_tokens: usage.prompt_tokens,
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .and_then(|details| details.cached_tokens)
                    .unwrap_or(0),
                output_tokens: usage.completion_tokens,
                reasoning_tokens: usage
                    .completion_tokens_details
                    .and_then(|details| details.reasoning_tokens)
                    .unwrap_or(0),
                requests: 1,
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
            .map(|tool_call| ToolCall {
                id: tool_call.id,
                name: tool_call.name,
                arguments: arguments_from_text(tool_call.arguments_text),
            })
            .collect();
        AssistantTurn {
            text: self.text,
            tool_calls,
            usage: self.usage,
            ..Default::default() // no reasoning or response id in this API
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
    request: &ModelRequest<'_>,
    text_output: &mut TextOutput,
) -> PyResult<AssistantTurn> {
    let response = client
        .post_json("/chat/completions", &build_request_body(request))
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
    use crate::model_api::test_helpers::{messages, request};
    use serde_json::Map;

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
            Usage {
                input_tokens: 10,
                output_tokens: 5,
                requests: 1,
                ..Default::default()
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
    fn cached_and_reasoning_tokens_are_read_from_usage_details() {
        let mut accumulator = ChunkAccumulator::default();
        accumulator
            .add_chunk(r#"{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":64},"completion_tokens_details":{"reasoning_tokens":30}}}"#)
            .unwrap();
        let usage = accumulator.into_turn().usage;
        assert_eq!((usage.cached_input_tokens, usage.reasoning_tokens), (64, 30));
    }

    #[test]
    fn missing_usage_counts_are_zero() {
        let mut accumulator = ChunkAccumulator::default();
        accumulator.add_chunk(r#"{"choices":[],"usage":{}}"#).unwrap();
        let expected = Usage {
            requests: 1,
            ..Default::default()
        };
        assert_eq!(accumulator.into_turn().usage, expected);
    }

    #[test]
    fn request_body_contains_history_and_tools() {
        let history = messages(json!([
            {"role": "user", "content": "add 1"},
            {"role": "assistant", "tool_calls": [{"id": "c", "name": "add", "arguments": {"a": 1}}],
             // Reasoning from the Responses API is never sent to this API.
             "reasoning": [{"id": "rs_1", "summary": ["thinking"], "encrypted_content": "secret"}]},
            {"role": "tool", "tool_call_id": "c", "name": "add", "content": "1"},
        ]));
        let mut parameters = Map::new();
        parameters.insert("temperature".into(), json!(0.2));
        parameters.insert("stream".into(), json!(false)); // must be overridden

        let body = build_request_body(&ModelRequest {
            system_prompt: "Be brief",
            ..request(&history, &parameters)
        });
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["temperature"], json!(0.2));
        assert_eq!(body["messages"][0], json!({"role": "system", "content": "Be brief"}));
        assert_eq!(
            body["messages"][2],
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "c", "type": "function", "function": {"name": "add", "arguments": "{\"a\":1}"}}],
            })
        );
        assert_eq!(
            body["messages"][3],
            json!({"role": "tool", "tool_call_id": "c", "content": "1"})
        );
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn images_and_files_become_content_parts() {
        let history = messages(json!([{"role": "user", "content": [
            {"type": "text", "text": "Compare"},
            {"type": "image", "url": "https://example.com/cat.png"},
            {"type": "file", "url": "data:application/pdf;base64,JVBERg==", "filename": "a.pdf"},
        ]}]));
        let body = build_request_body(&request(&history, &Map::new()));
        assert_eq!(
            body["messages"][0]["content"],
            json!([
                {"type": "text", "text": "Compare"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERg==", "filename": "a.pdf"}},
            ])
        );
    }
}
