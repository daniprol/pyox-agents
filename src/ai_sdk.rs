//! A run as a Vercel AI SDK "UI message stream": what `useChat` (and
//! assistant-ui) read. Python: `StreamingResponse(run.ai_sdk(), headers=ai_sdk.HEADERS)`.
//!
//! Each run event becomes one or more chunks, each sent as `data: {json}`:
//!
//! | Run event                     | Chunks                                                     |
//! |-------------------------------|------------------------------------------------------------|
//! | (the stream starts)           | `start`                                                    |
//! | `TextDelta`                   | `text-start` (once per text part), `text-delta`            |
//! | `ToolStarted`                 | `tool-input-available`                                     |
//! | `ToolFinished`                | `tool-output-available`, or `tool-output-error`            |
//! | `InputRequested`              | `data-pyox-request` (and `tool-approval-request` for approvals) |
//! | `StateChanged`                | `data-state`: the whole state (it replaces the last one)   |
//! | a subagent's events           | `data-subagent` (see below)                                |
//! | `RunFinished`                 | `finish`, then `[DONE]`                                    |
//! | a pyox-agents error           | `error`, then `[DONE]`                                     |
//!
//! Model turns are AI SDK "steps": `start-step` ... `finish-step`. Tools run
//! after the model's turn ends, so the first text or tool call after a tool
//! result starts a new step.
//!
//! `data-pyox-request` tells the frontend that a tool call waits for a person
//! (`waiting`, `kind`, `data`, `metadata`), and comes back in the next request
//! so the server knows which tool outputs are answers (`pyox_agents.ai_sdk.parse`).
//! When the call resumes, the part is replaced by one with `waiting: false`.
//! Approvals need AI SDK 6 or newer. There is no spec: this follows the
//! chunk schema of the `ai` package (versions 6 and 7).
//!
//! A subagent's work is shown in a `data-subagent` part whose id is the id of
//! the tool call running it: `{toolCallId, name, task, status, result, parts}`,
//! where `status` is `running`, `waiting`, `done` or `error`, and `parts` are
//! UI message parts (`text`, `dynamic-tool`). A newer one replaces the older
//! one; it is sent once per batch of events, not for every piece of text.
//! When a subagent's tool waits for the person, the subagent's tool call waits:
//! its `data-pyox-request` has the request's own id in `requestId`, and an
//! approval's `approvalId` is that id too.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::events::AgentEvent;
use crate::messages::RequestKind;
use crate::run::{new_id, write_sse};

#[derive(Default)]
pub struct AiSdkEncoder {
    /// The id of the text part being streamed, if one is open.
    text_id: Option<String>,
    /// A step (a model turn) is open.
    in_step: bool,
    /// Tool results came after the step's model output: the next model output is a new step.
    after_tools: bool,
    /// The model wrote something in this stream. Tool results before that are
    /// calls the run resumed with the user's answers.
    model_output_seen: bool,
    /// The `data-subagent` part of each subagent, by its id.
    subagents: HashMap<String, Value>,
    /// The subagents whose part changed since it was last sent.
    changed_subagents: Vec<String>,
}

impl AiSdkEncoder {
    pub fn start(&mut self, out: &mut String) {
        // No `messageId`: after answers, the client then continues its last assistant message.
        write_sse(out, &json!({"type": "start"}));
    }

    pub fn add_event(&mut self, event: &AgentEvent, out: &mut String) {
        if let Some(subagent_id) = event.subagent_id() {
            return self.add_subagent_event(subagent_id, event, out);
        }
        match event {
            AgentEvent::TextDelta(delta) => {
                if delta.text.is_empty() {
                    return;
                }
                self.start_model_output(out);
                let text_id = match &self.text_id {
                    Some(text_id) => text_id.clone(),
                    None => {
                        let text_id = new_id("text");
                        write_sse(out, &json!({"type": "text-start", "id": text_id}));
                        self.text_id = Some(text_id.clone());
                        text_id
                    }
                };
                write_sse(out, &json!({"type": "text-delta", "id": text_id, "delta": delta.text}));
            }
            AgentEvent::ToolStarted(call) => {
                self.start_model_output(out);
                self.end_text(out);
                write_sse(
                    out,
                    &json!({
                        "type": "tool-input-available",
                        "toolCallId": call.call_id,
                        "toolName": call.name,
                        "input": call.arguments,
                    }),
                );
            }
            AgentEvent::ToolFinished(result) => {
                self.end_text(out);
                if !self.in_step {
                    // A resumed run starts with tool results.
                    write_sse(out, &json!({"type": "start-step"}));
                    self.in_step = true;
                }
                self.after_tools = true;
                if !self.model_output_seen {
                    // A resumed call: it no longer waits (see `data-pyox-request`).
                    write_sse(
                        out,
                        &waiting_part(&result.call_id, json!({"toolCallId": result.call_id, "waiting": false})),
                    );
                }
                if result.from_app {
                    return; // the client gave this result: it already has it
                }
                let chunk = if result.is_error {
                    json!({"type": "tool-output-error", "toolCallId": result.call_id, "errorText": result.result})
                } else {
                    json!({"type": "tool-output-available", "toolCallId": result.call_id, "output": result.result})
                };
                write_sse(out, &chunk);
            }
            AgentEvent::InputRequested(request) => {
                let request_data = json!({
                    "toolCallId": request.call_id,
                    "toolName": request.tool_name,
                    "waiting": true,
                    "kind": request.kind.name(),
                    "key": request.key,
                    "data": request.data,
                    "metadata": request.metadata,
                });
                write_sse(out, &waiting_part(&request.call_id, request_data));
                if request.kind == RequestKind::Approval {
                    write_sse(
                        out,
                        &json!({
                            "type": "tool-approval-request",
                            "approvalId": request.call_id,
                            "toolCallId": request.call_id,
                        }),
                    );
                }
            }
            AgentEvent::StateChanged(changed) => {
                write_sse(
                    out,
                    &json!({"type": "data-state", "id": "state", "data": changed.state}),
                );
            }
            AgentEvent::SubagentStarted(started) => {
                let part = json!({
                    "toolCallId": started.call_id,
                    "name": started.name,
                    "task": started.task,
                    "status": "running",
                    "parts": [],
                });
                self.subagents.insert(started.call_id.clone(), part);
                self.subagent_changed(&started.call_id);
            }
            AgentEvent::SubagentFinished(finished) => {
                if let Some(part) = self.subagents.get_mut(&finished.call_id) {
                    part["status"] = json!(match (finished.is_error, finished.waiting) {
                        (true, _) => "error",
                        (false, true) => "waiting",
                        (false, false) => "done",
                    });
                    part["result"] = json!(finished.result);
                    self.subagent_changed(&finished.call_id);
                }
            }
            AgentEvent::RunFinished(finished) => {
                self.end_batch(out);
                self.end_text(out);
                self.end_step(out);
                let finish_reason = if finished.pending.is_empty() {
                    "stop"
                } else {
                    "tool-calls"
                };
                write_sse(out, &json!({"type": "finish", "finishReason": finish_reason}));
                out.push_str("data: [DONE]\n\n");
            }
        }
    }

    pub fn add_error(&mut self, message: &str, out: &mut String) {
        self.end_batch(out);
        self.end_text(out);
        write_sse(out, &json!({"type": "error", "errorText": message}));
        out.push_str("data: [DONE]\n\n");
    }

    /// Sends the `data-subagent` parts that changed in this batch of events.
    pub fn end_batch(&mut self, out: &mut String) {
        for subagent_id in std::mem::take(&mut self.changed_subagents) {
            let part = &self.subagents[&subagent_id];
            write_sse(out, &json!({"type": "data-subagent", "id": subagent_id, "data": part}));
        }
    }

    fn subagent_changed(&mut self, subagent_id: &str) {
        if !self.changed_subagents.iter().any(|id| id == subagent_id) {
            self.changed_subagents.push(subagent_id.to_string());
        }
    }

    /// A subagent's event goes into its `data-subagent` part. Only its
    /// requests are sent at once, on the tool call running the subagent.
    fn add_subagent_event(&mut self, subagent_id: &str, event: &AgentEvent, out: &mut String) {
        if let AgentEvent::InputRequested(request) = event {
            let request_data = json!({
                "toolCallId": subagent_id,
                "requestId": request.call_id,
                "toolName": request.tool_name,
                "waiting": true,
                "kind": request.kind.name(),
                "key": request.key,
                "data": request.data,
                "metadata": request.metadata,
            });
            write_sse(out, &waiting_part(subagent_id, request_data));
            if request.kind == RequestKind::Approval {
                write_sse(
                    out,
                    &json!({"type": "tool-approval-request", "approvalId": request.call_id, "toolCallId": subagent_id}),
                );
            }
            return;
        }
        let Some(Value::Array(parts)) = self.subagents.get_mut(subagent_id).map(|part| &mut part["parts"]) else {
            return; // its `SubagentStarted` came first, so this does not happen
        };
        match event {
            AgentEvent::TextDelta(delta) => match parts.last_mut() {
                Some(last) if last["type"] == "text" => {
                    let text = format!("{}{}", last["text"].as_str().unwrap_or_default(), delta.text);
                    last["text"] = json!(text);
                }
                _ => parts.push(json!({"type": "text", "text": delta.text})),
            },
            AgentEvent::ToolStarted(call) => parts.push(json!({
                "type": "dynamic-tool",
                "toolCallId": call.call_id,
                "toolName": call.name,
                "state": "input-available",
                "input": call.arguments,
            })),
            AgentEvent::ToolFinished(result) => {
                let Some(part) = parts
                    .iter_mut()
                    .find(|part| part["toolCallId"] == json!(result.call_id))
                else {
                    return;
                };
                if result.is_error {
                    part["state"] = json!("output-error");
                    part["errorText"] = json!(result.result);
                } else {
                    part["state"] = json!("output-available");
                    part["output"] = json!(result.result);
                }
            }
            // The state is the run's, whoever changed it.
            AgentEvent::StateChanged(changed) => {
                write_sse(
                    out,
                    &json!({"type": "data-state", "id": "state", "data": changed.state}),
                );
                return;
            }
            _ => return,
        }
        self.subagent_changed(subagent_id);
    }

    /// Opens a step for model output, closing the previous one if tools ran since.
    fn start_model_output(&mut self, out: &mut String) {
        self.model_output_seen = true;
        if self.in_step && !self.after_tools {
            return;
        }
        self.end_step(out);
        write_sse(out, &json!({"type": "start-step"}));
        self.in_step = true;
        self.after_tools = false;
    }

    fn end_step(&mut self, out: &mut String) {
        if self.in_step {
            write_sse(out, &json!({"type": "finish-step"}));
            self.in_step = false;
        }
    }

    fn end_text(&mut self, out: &mut String) {
        if let Some(text_id) = self.text_id.take() {
            write_sse(out, &json!({"type": "text-end", "id": text_id}));
        }
    }
}

/// The `data-pyox-request` part of a tool call: whether it waits for the user,
/// and for what. Its id is the call id, so a newer one replaces the older one
/// in the client's message.
fn waiting_part(call_id: &str, data: serde_json::Value) -> serde_json::Value {
    json!({"type": "data-pyox-request", "id": call_id, "data": data})
}
