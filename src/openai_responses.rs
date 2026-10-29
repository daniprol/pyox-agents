//! Adapter for the OpenAI Responses API (`POST /responses`).
//!
//! Like `openai_chat.rs`, it translates in both directions:
//! - our `Message` history  ->  the `input` items of the request
//! - the streamed events    ->  `TextDelta` events + one finished `AssistantTurn`
//!
//! What this API adds over Chat Completions:
//! - Reasoning. We ask for it encrypted (`include: ["reasoning.encrypted_content"]`),
//!   keep it in the assistant message and send it back unchanged next time.
//! - Stored responses. With `store: true` OpenAI keeps each response, and a
//!   request can name the one it continues (`previous_response_id`) and send
//!   only the newer messages.
//!
//! How our history becomes `input` items:
//!
//! ```text
//! user message       ->  {"role": "user", "content": ...}
//! assistant message  ->  its reasoning items, then {"role": "assistant", "content": "..."},
//!                        then one {"type": "function_call", ...} per tool call
//! tool message       ->  {"type": "function_call_output", "call_id": ..., "output": ...}
//! ```

use futures::StreamExt;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::client::ClientSettings;
use crate::errors::{ApiError, InvalidResponseError, error_for_http_failure, error_for_http_status};
use crate::messages::{
    ContentPart, Message, Reasoning, TokenUsage, ToolCall, UserContent, arguments_as_text, arguments_from_text,
};
use crate::model_api::{AssistantTurn, ModelRequest, ResponsesOptions};
use crate::sse::SseDecoder;
use crate::text_chunks::TextOutput;

// ---------------------------------------------------------------------------
// Request: our history -> Responses JSON
// ---------------------------------------------------------------------------

/// Builds the request body. `model_parameters` (reasoning effort, ...) go
/// first so the fields this library controls always win.
///
/// With a `previous_response_id`, only the messages after that response are
/// sent: OpenAI already has the rest.
pub fn build_request_body(
    request: &ModelRequest<'_>,
    options: ResponsesOptions,
    previous_response_id: Option<&str>,
) -> Value {
    let mut body = request.model_parameters.clone();
    body.insert("model".into(), json!(request.model));
    // The system prompt. A stored response does not carry it over, so every request sends it.
    if !request.system_prompt.is_empty() {
        body.insert("instructions".into(), json!(request.system_prompt));
    }

    let mut messages_to_send = request.messages;
    if let Some(response_id) = previous_response_id {
        body.insert("previous_response_id".into(), json!(response_id));
        // When the response is not in our messages (it came from
        // `execute(previous_response_id=...)`), all our messages are new to OpenAI.
        if let Some(newer_messages) = messages_after_response(request.messages, response_id) {
            messages_to_send = newer_messages;
        }
    }
    body.insert(
        "input".into(),
        Value::Array(input_items(request.model, messages_to_send)),
    );
    body.insert("stream".into(), json!(true));
    body.insert("store".into(), json!(options.store));

    if options.encrypted_reasoning {
        add_to_include(&mut body, "reasoning.encrypted_content");
    }

    if !request.tools.is_empty() {
        let tool_schemas = request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    // This API defaults to `strict: true`, which only accepts schemas where
                    // every property is required. Our tools allow optional arguments.
                    "strict": false,
                })
            })
            .collect();
        body.insert("tools".into(), Value::Array(tool_schemas));
    }
    Value::Object(body)
}

/// The messages that come after the assistant message with this response id,
/// or `None` if no message has it.
fn messages_after_response<'a>(messages: &'a [Message], response_id: &str) -> Option<&'a [Message]> {
    let position = messages
        .iter()
        .position(|message| matches!(message, Message::Assistant { response_id: Some(id), .. } if id == response_id))?; // `?` on an `Option`: return `None` right away if it is `None`
    Some(&messages[position + 1..])
}

/// Adds `value` to the body's `include` list, keeping what the user put there.
fn add_to_include(body: &mut Map<String, Value>, value: &str) {
    let include = body.entry("include").or_insert_with(|| json!([]));
    if let Value::Array(values) = include
        && !values.iter().any(|existing| existing == value)
    {
        values.push(json!(value));
    }
}

fn input_items(model: &str, messages: &[Message]) -> Vec<Value> {
    let mut items = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            Message::User { content } => items.push(json!({"role": "user", "content": user_content(content)})),
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
                model: message_model,
                ..
            } => {
                // Reasoning only means something to the model that wrote it. And a
                // reasoning item must be followed by what it led to (text or a tool
                // call), or the API rejects the request.
                let same_model = message_model.as_deref() == Some(model);
                let leads_somewhere = !content.is_empty() || !tool_calls.is_empty();
                if same_model && leads_somewhere {
                    items.extend(reasoning.iter().filter_map(reasoning_item));
                }
                if !content.is_empty() {
                    items.push(json!({"role": "assistant", "content": content}));
                }
                for call in tool_calls {
                    items.push(json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": arguments_as_text(&call.arguments),
                    }));
                }
            }
            Message::Tool {
                tool_call_id, content, ..
            } => items.push(json!({"type": "function_call_output", "call_id": tool_call_id, "output": content})),
        }
    }
    items
}

/// The reasoning item exactly as the API returned it. Without the encrypted
/// content there is nothing the API accepts back, so the item is skipped.
fn reasoning_item(reasoning: &Reasoning) -> Option<Value> {
    let encrypted_content = reasoning.encrypted_content.as_ref()?;
    let summary: Vec<Value> = reasoning
        .summary
        .iter()
        .map(|text| json!({"type": "summary_text", "text": text}))
        .collect();
    Some(json!({
        "type": "reasoning",
        "id": reasoning.id,
        "summary": summary,
        "encrypted_content": encrypted_content,
    }))
}

/// Text stays a plain string; parts become the API's list of input parts.
fn user_content(content: &UserContent) -> Value {
    match content {
        UserContent::Text(text) => json!(text),
        UserContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => json!({"type": "input_text", "text": text}),
                ContentPart::Image { url } => json!({"type": "input_image", "image_url": url}),
                // The file's bytes in a `data:` URL, or a web address the API downloads.
                ContentPart::File { url, filename } if url.starts_with("data:") => {
                    json!({"type": "input_file", "filename": filename, "file_data": url})
                }
                ContentPart::File { url, .. } => json!({"type": "input_file", "file_url": url}),
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Response: streamed events -> AssistantTurn
// ---------------------------------------------------------------------------

// Each streamed event is a JSON object whose `"type"` says what it is. We only
// need a few of them:
//
// - `response.output_text.delta`: a piece of text, forwarded right away.
// - `response.completed` (or `response.incomplete`, when the answer was cut
//   short): the whole response. We read the turn from it, since it holds
//   everything at once (text, tool calls, reasoning, usage).
// - `response.failed` and `error`: the request failed.
//
// Fields we do not need are ignored by serde.

#[derive(Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")] // `type` is a Rust keyword, so the field has another name
    kind: String,
    delta: Option<String>,
    /// Read as plain JSON first: only the finished response is parsed further.
    response: Option<Value>,
}

#[derive(Deserialize)]
struct FinishedResponse {
    id: String,
    #[serde(default)]
    output: Vec<OutputItem>,
    usage: Option<ResponseUsage>,
}

/// One item of the response's `output` list. `#[serde(tag = "type")]` picks
/// the variant from the item's `"type"` field (`"message"`, `"function_call"`,
/// ...); `#[serde(other)]` catches every type we do not use (e.g. web search calls).
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputItem {
    Message {
        #[serde(default)]
        content: Vec<MessagePart>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<SummaryPart>,
        encrypted_content: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MessagePart {
    OutputText {
        text: String,
    },
    /// The model declined to answer; we show its explanation as the text.
    Refusal {
        refusal: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct SummaryPart {
    text: String,
}

#[derive(Deserialize)]
struct ResponseUsage {
    input_tokens: u64,
    output_tokens: u64,
    input_tokens_details: Option<InputTokensDetails>,
}

#[derive(Deserialize)]
struct InputTokensDetails {
    cached_tokens: Option<u64>,
}

/// Reads the events of one response until it is complete.
#[derive(Default)]
pub struct EventReader {
    /// Set by `response.completed`; `None` until then.
    turn: Option<AssistantTurn>,
}

impl EventReader {
    /// Reads one event (the JSON text of an SSE event). Returns the new text, if any.
    pub fn add_event(&mut self, event_json: &str) -> PyResult<Option<String>> {
        let event: StreamEvent = serde_json::from_str(event_json)
            .map_err(|error| InvalidResponseError::new_err(format!("{error}: {event_json}")))?;

        match event.kind.as_str() {
            "response.output_text.delta" => Ok(event.delta.filter(|text| !text.is_empty())),
            "response.completed" | "response.incomplete" => {
                let response: FinishedResponse = event
                    .response
                    .and_then(|response| serde_json::from_value(response).ok())
                    .ok_or_else(|| InvalidResponseError::new_err(format!("Invalid response: {event_json}")))?;
                self.turn = Some(turn_from_response(response));
                Ok(None)
            }
            "response.failed" | "error" => Err(ApiError::new_err(event_json.to_string())),
            _ => Ok(None), // events we do not need
        }
    }

    /// The finished turn, or an error if the stream ended before the response was complete.
    pub fn into_turn(self) -> PyResult<AssistantTurn> {
        self.turn
            .ok_or_else(|| InvalidResponseError::new_err("The response stream ended unexpectedly"))
    }
}

fn turn_from_response(response: FinishedResponse) -> AssistantTurn {
    let mut turn = AssistantTurn {
        response_id: Some(response.id),
        ..Default::default()
    };
    for item in response.output {
        match item {
            OutputItem::Message { content } => {
                for part in content {
                    match part {
                        MessagePart::OutputText { text } | MessagePart::Refusal { refusal: text } => {
                            turn.text.push_str(&text)
                        }
                        MessagePart::Other => {}
                    }
                }
            }
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => turn.tool_calls.push(ToolCall {
                id: call_id,
                name,
                arguments: arguments_from_text(arguments),
            }),
            OutputItem::Reasoning {
                id,
                summary,
                encrypted_content,
            } => turn.reasoning.push(Reasoning {
                id,
                summary: summary.into_iter().map(|part| part.text).collect(),
                encrypted_content,
            }),
            OutputItem::Other => {}
        }
    }
    if let Some(usage) = response.usage {
        turn.usage = TokenUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cached_input_tokens: usage
                .input_tokens_details
                .and_then(|details| details.cached_tokens)
                .unwrap_or(0),
        };
    }
    turn
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

/// Sends one request and streams the answer. Each piece of text is sent to
/// `text_output` as soon as it arrives.
pub async fn stream_response(
    client: &ClientSettings,
    request: &ModelRequest<'_>,
    options: ResponsesOptions,
    text_output: &mut TextOutput,
) -> PyResult<AssistantTurn> {
    let mut previous_response_id = request.previous_response_id;
    loop {
        let request_body = build_request_body(request, options, previous_response_id);
        let response = client
            .post_json("/responses", &request_body)
            .await
            .map_err(error_for_http_failure)?;
        let status = response.status();
        if status.is_success() {
            return read_events(response, text_output).await;
        }

        let response_body = response.text().await.unwrap_or_default();
        // The stored response we continue from is gone (deleted, or older than
        // OpenAI keeps them). If it is in our messages, we have the whole
        // conversation: send all of it instead. This happens at most once,
        // since the second request has no `previous_response_id`.
        let have_whole_history =
            previous_response_id.is_some_and(|id| messages_after_response(request.messages, id).is_some());
        if have_whole_history && response_body.contains("previous_response_not_found") {
            previous_response_id = None;
            continue;
        }
        return Err(error_for_http_status(status.as_u16(), &response_body));
    }
}

async fn read_events(response: reqwest::Response, text_output: &mut TextOutput) -> PyResult<AssistantTurn> {
    let mut sse_decoder = SseDecoder::default();
    let mut event_reader = EventReader::default();
    let mut byte_stream = response.bytes_stream();

    while let Some(network_chunk) = byte_stream.next().await {
        let network_chunk = network_chunk.map_err(error_for_http_failure)?;
        for event_data in sse_decoder.push(&network_chunk) {
            // OpenAI does not send `[DONE]` here, but some compatible servers do.
            if event_data == "[DONE]" {
                continue;
            }
            if let Some(text) = event_reader.add_event(&event_data)? {
                text_output.push(&text).await;
            }
        }
    }
    if let Some(event_data) = sse_decoder.finish()
        && event_data != "[DONE]"
    {
        event_reader.add_event(&event_data)?;
    }
    event_reader.into_turn()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_api::test_helpers::{messages, request};

    const STATELESS: ResponsesOptions = ResponsesOptions {
        store: false,
        encrypted_reasoning: true,
    };

    /// An assistant message from model "m" that reasoned, then called a tool.
    fn conversation_with_reasoning() -> Vec<Message> {
        messages(json!([
            {"role": "user", "content": "add 1"},
            {"role": "assistant", "model": "m", "response_id": "resp_1",
             "reasoning": [{"id": "rs_1", "summary": ["Use add."], "encrypted_content": "secret"}],
             "tool_calls": [{"id": "call_1", "name": "add", "arguments": {"a": 1}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "add", "content": "1"},
        ]))
    }

    #[test]
    fn history_becomes_input_items() {
        let history = conversation_with_reasoning();
        let mut parameters = Map::new();
        parameters.insert("include".into(), json!(["web_search_call.results"]));
        parameters.insert("stream".into(), json!(false)); // must be overridden

        let body = build_request_body(
            &ModelRequest {
                system_prompt: "Be brief",
                ..request(&history, &parameters)
            },
            STATELESS,
            None,
        );

        assert_eq!(body["instructions"], json!("Be brief"));
        assert_eq!((&body["stream"], &body["store"]), (&json!(true), &json!(false)));
        // Our include is added to the user's.
        assert_eq!(
            body["include"],
            json!(["web_search_call.results", "reasoning.encrypted_content"])
        );
        assert_eq!(
            body["input"],
            json!([
                {"role": "user", "content": "add 1"},
                {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Use add."}],
                 "encrypted_content": "secret"},
                {"type": "function_call", "call_id": "call_1", "name": "add", "arguments": "{\"a\":1}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "1"},
            ])
        );
        assert!(body.get("previous_response_id").is_none());
    }

    #[test]
    fn reasoning_is_not_sent_to_another_model() {
        let history = conversation_with_reasoning();
        let parameters = Map::new();
        let body = build_request_body(
            &ModelRequest {
                model: "another-model",
                ..request(&history, &parameters)
            },
            STATELESS,
            None,
        );
        let item_types: Vec<_> = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.get("type").cloned().unwrap_or(json!("message")))
            .collect();
        assert_eq!(
            item_types,
            vec![json!("message"), json!("function_call"), json!("function_call_output")]
        );
    }

    #[test]
    fn previous_response_id_sends_only_newer_messages() {
        let mut history = conversation_with_reasoning();
        history.push(Message::User {
            content: "and 2?".into(),
        });
        let parameters = Map::new();
        let options = ResponsesOptions {
            store: true,
            encrypted_reasoning: false,
        };

        let body = build_request_body(&request(&history, &parameters), options, Some("resp_1"));
        assert_eq!(body["previous_response_id"], json!("resp_1"));
        assert_eq!(
            body["input"],
            json!([
                {"type": "function_call_output", "call_id": "call_1", "output": "1"},
                {"role": "user", "content": "and 2?"},
            ])
        );
        assert!(body.get("include").is_none());

        // A response from outside the history (`execute(previous_response_id=...)`):
        // every message is new (2 user messages, the reasoning, the call and its output).
        let body = build_request_body(&request(&history, &parameters), options, Some("resp_older"));
        assert_eq!(body["input"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn images_and_files_become_input_parts() {
        let history = messages(json!([{"role": "user", "content": [
            {"type": "text", "text": "Compare"},
            {"type": "image", "url": "https://example.com/cat.png"},
            {"type": "file", "url": "data:application/pdf;base64,JVBERg==", "filename": "a.pdf"},
            {"type": "file", "url": "https://example.com/b.pdf", "filename": "b.pdf"},
        ]}]));
        let parameters = Map::new();
        let body = build_request_body(&request(&history, &parameters), STATELESS, None);
        assert_eq!(
            body["input"][0]["content"],
            json!([
                {"type": "input_text", "text": "Compare"},
                {"type": "input_image", "image_url": "https://example.com/cat.png"},
                {"type": "input_file", "filename": "a.pdf", "file_data": "data:application/pdf;base64,JVBERg=="},
                {"type": "input_file", "file_url": "https://example.com/b.pdf"},
            ])
        );
    }

    #[test]
    fn the_completed_response_becomes_the_turn() {
        let mut reader = EventReader::default();
        let events = [
            r#"{"type":"response.created","response":{"id":"resp_9","output":[]}}"#,
            r#"{"type":"response.output_text.delta","delta":"Hi"}"#,
            r#"{"type":"response.output_text.delta","delta":""}"#,
            r#"{"type":"response.completed","response":{"id":"resp_9","output":[
                {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Greet."}],"encrypted_content":"secret"},
                {"type":"message","content":[{"type":"output_text","text":"Hi"}]},
                {"type":"web_search_call","id":"ws_1"},
                {"type":"function_call","call_id":"call_1","name":"add","arguments":"{\"a\":1}"}
              ],"usage":{"input_tokens":100,"output_tokens":5,"input_tokens_details":{"cached_tokens":64}}}}"#,
        ];
        let texts: Vec<_> = events.iter().map(|event| reader.add_event(event).unwrap()).collect();
        assert_eq!(texts, vec![None, Some("Hi".to_string()), None, None]);

        assert_eq!(
            reader.into_turn().unwrap(),
            AssistantTurn {
                text: "Hi".into(),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "add".into(),
                    arguments: json!({"a": 1}),
                }],
                reasoning: vec![Reasoning {
                    id: "rs_1".into(),
                    summary: vec!["Greet.".into()],
                    encrypted_content: Some("secret".into()),
                }],
                response_id: Some("resp_9".into()),
                usage: TokenUsage {
                    input_tokens: 100,
                    output_tokens: 5,
                    cached_input_tokens: 64,
                },
            }
        );
    }

    #[test]
    fn a_stream_without_its_completed_event_is_an_error() {
        let mut reader = EventReader::default();
        reader
            .add_event(r#"{"type":"response.output_text.delta","delta":"Hi"}"#)
            .unwrap();
        assert!(reader.into_turn().is_err());
    }
}
