//! A run as an AG-UI event stream (CopilotKit and other AG-UI clients).
//! Python: `StreamingResponse(run.ag_ui(thread_id, run_id), headers=ag_ui.HEADERS)`.
//!
//! Each run event becomes one or more AG-UI events, each sent as `data: {json}`:
//!
//! | Run event          | AG-UI events                                                         |
//! |--------------------|----------------------------------------------------------------------|
//! | (stream starts)    | `RUN_STARTED`                                                        |
//! | `TextDelta`        | `TEXT_MESSAGE_START` (once per message), `TEXT_MESSAGE_CONTENT`      |
//! | `ToolStarted`      | `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`                 |
//! | `ToolFinished`     | `TOOL_CALL_RESULT`                                                   |
//! | `InputRequested`   | nothing yet: listed in `RUN_FINISHED`                                |
//! | `StateChanged`     | `STATE_DELTA` (a JSON Patch on the state the client sent)            |
//! | `SubagentStarted`  | `SUBAGENT_STARTED`                                                   |
//! | `SubagentFinished` | `SUBAGENT_FINISHED` (`suspended` while it waits), or `SUBAGENT_ERROR` |
//! | `RunFinished`      | `RUN_FINISHED`, with an `interrupt` outcome when approvals or questions wait |
//! | a pyox-agents error | `RUN_ERROR`                                                         |
//!
//! A subagent's events carry `subagentRunId` (its id: the id of the tool call
//! running it), and its messages and tool calls have their own ids.
//!
//! Waiting calls, as AG-UI 1.0 expects them:
//! - an approval or a `ctx.ask` question is an *interrupt* whose id is the
//!   tool call id. The client resumes with `resume: [{interruptId, status, payload}]`;
//!   an approval's payload is `{approved, editedArgs?, reason?}`.
//! - a frontend tool is a *pending tool call*: the client runs it and sends
//!   the result as a tool message in the next run.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::events::AgentEvent;
use crate::messages::RequestKind;
use crate::run::{new_id, write_sse};

pub struct AgUiEncoder {
    thread_id: String,
    run_id: String,
    /// The current message of the main agent (key `None`) and of each subagent.
    messages: HashMap<Option<String>, OpenMessage>,
    /// Approvals and questions waiting, for the `RUN_FINISHED` outcome.
    interrupts: Vec<Value>,
    /// Frontend tool calls waiting for the client.
    pending_tool_call_ids: Vec<String>,
}

/// The assistant message an agent is writing in its current model turn
/// (its text and tool calls belong to it).
#[derive(Default)]
struct OpenMessage {
    id: Option<String>,
    /// Its text was started (`TEXT_MESSAGE_START`) and not ended yet.
    text_open: bool,
    /// Tool results came after the turn's output: the next output is a new message.
    after_tools: bool,
}

impl AgUiEncoder {
    pub fn new(thread_id: String, run_id: String) -> Self {
        AgUiEncoder {
            thread_id,
            run_id,
            messages: HashMap::new(),
            interrupts: Vec::new(),
            pending_tool_call_ids: Vec::new(),
        }
    }

    pub fn start(&mut self, out: &mut String) {
        write_sse(
            out,
            &json!({"type": "RUN_STARTED", "threadId": self.thread_id, "runId": self.run_id}),
        );
    }

    pub fn add_event(&mut self, event: &AgentEvent, out: &mut String) {
        let subagent_id = event.subagent_id().map(str::to_string);
        // Every event of a subagent says so.
        let write = |out: &mut String, mut data: Value| {
            if let Some(subagent_id) = &subagent_id {
                data["subagentRunId"] = json!(subagent_id);
            }
            write_sse(out, &data);
        };
        match event {
            AgentEvent::TextDelta(delta) => {
                if delta.text.is_empty() {
                    return;
                }
                let message = self.message_of(&subagent_id, out);
                let message_id = message.id.clone();
                if !message.text_open {
                    message.text_open = true;
                    write(
                        out,
                        json!({"type": "TEXT_MESSAGE_START", "messageId": message_id, "role": "assistant"}),
                    );
                }
                write(
                    out,
                    json!({"type": "TEXT_MESSAGE_CONTENT", "messageId": message_id, "delta": delta.text}),
                );
            }
            AgentEvent::ToolStarted(call) => {
                let message_id = self.message_of(&subagent_id, out).id.clone();
                self.end_text(&subagent_id, out);
                write(
                    out,
                    json!({
                        "type": "TOOL_CALL_START",
                        "toolCallId": call.call_id,
                        "toolCallName": call.name,
                        "parentMessageId": message_id,
                    }),
                );
                write(
                    out,
                    json!({"type": "TOOL_CALL_ARGS", "toolCallId": call.call_id, "delta": call.arguments.to_string()}),
                );
                write(out, json!({"type": "TOOL_CALL_END", "toolCallId": call.call_id}));
            }
            AgentEvent::ToolFinished(result) => {
                self.end_text(&subagent_id, out);
                self.messages.entry(subagent_id.clone()).or_default().after_tools = true;
                if result.from_app {
                    return; // the client gave this result: it already has it
                }
                let mut event = json!({
                    "type": "TOOL_CALL_RESULT",
                    "messageId": new_id("msg"),
                    "toolCallId": result.call_id,
                    "content": result.result,
                    "role": "tool",
                });
                // `ToolReturn(metadata=...)`, for the client only (AG-UI metadata is an object).
                if result.metadata.is_object() {
                    event["metadata"] = result.metadata.clone();
                }
                write(out, event);
            }
            AgentEvent::InputRequested(request) => match request.kind {
                RequestKind::Frontend => self.pending_tool_call_ids.push(request.call_id.clone()),
                RequestKind::Approval | RequestKind::Input => {
                    let mut interrupt = json!({
                        "id": request.call_id,
                        "reason": request.kind.name(),
                        "toolCallId": request.call_id,
                        "metadata": {"data": request.data, "metadata": request.metadata},
                    });
                    if let Value::String(question) = &request.data {
                        interrupt["message"] = json!(question);
                    }
                    if request.kind == RequestKind::Approval {
                        interrupt["responseSchema"] = approval_schema();
                    }
                    if let Some(subagent_id) = &subagent_id {
                        interrupt["subagentRunId"] = json!(subagent_id);
                    }
                    self.interrupts.push(interrupt);
                }
            },
            AgentEvent::StateChanged(changed) => {
                write(out, json!({"type": "STATE_DELTA", "delta": changed.delta}));
            }
            AgentEvent::SubagentStarted(started) => {
                let mut event = json!({
                    "type": "SUBAGENT_STARTED",
                    "subagentRunId": started.call_id,
                    "name": started.name,
                    "parentToolCallId": started.call_id,
                });
                if let Some(parent_id) = &subagent_id {
                    event["parentSubagentRunId"] = json!(parent_id);
                }
                write_sse(out, &event);
            }
            AgentEvent::SubagentFinished(finished) => {
                let id = Some(finished.call_id.clone());
                self.end_text(&id, out);
                self.messages.remove(&id);
                let event = if finished.is_error {
                    json!({"type": "SUBAGENT_ERROR", "subagentRunId": finished.call_id, "message": finished.result})
                } else if finished.waiting {
                    let interrupt_ids: Vec<&Value> = self
                        .interrupts
                        .iter()
                        .filter(|interrupt| interrupt["subagentRunId"] == json!(finished.call_id))
                        .map(|interrupt| &interrupt["id"])
                        .collect();
                    json!({
                        "type": "SUBAGENT_FINISHED",
                        "subagentRunId": finished.call_id,
                        "outcome": {"type": "suspended", "interruptIds": interrupt_ids},
                    })
                } else {
                    json!({
                        "type": "SUBAGENT_FINISHED",
                        "subagentRunId": finished.call_id,
                        "result": finished.result,
                        "outcome": {"type": "success"},
                    })
                };
                write_sse(out, &event);
            }
            AgentEvent::RunFinished(_) => {
                self.end_text(&None, out);
                let mut event = json!({"type": "RUN_FINISHED", "threadId": self.thread_id, "runId": self.run_id});
                if !self.interrupts.is_empty() {
                    event["outcome"] = json!({"type": "interrupt", "interrupts": self.interrupts});
                } else if !self.pending_tool_call_ids.is_empty() {
                    event["outcome"] = json!({"type": "success", "pendingToolCallIds": self.pending_tool_call_ids});
                }
                write_sse(out, &event);
            }
        }
    }

    pub fn add_error(&mut self, message: &str, code: &str, out: &mut String) {
        write_sse(out, &json!({"type": "RUN_ERROR", "message": message, "code": code}));
    }

    /// The current message of an agent, with an id; a new one after tool results.
    fn message_of(&mut self, subagent_id: &Option<String>, out: &mut String) -> &mut OpenMessage {
        if self
            .messages
            .get(subagent_id)
            .is_some_and(|message| message.after_tools)
        {
            self.end_text(subagent_id, out);
            self.messages.remove(subagent_id);
        }
        let message = self.messages.entry(subagent_id.clone()).or_default();
        message.id.get_or_insert_with(|| new_id("msg"));
        message
    }

    /// AG-UI clients reject a run that finishes with a text message still open.
    fn end_text(&mut self, subagent_id: &Option<String>, out: &mut String) {
        let Some(message) = self.messages.get_mut(subagent_id) else {
            return;
        };
        if message.text_open
            && let Some(message_id) = &message.id
        {
            let mut event = json!({"type": "TEXT_MESSAGE_END", "messageId": message_id});
            if let Some(subagent_id) = subagent_id {
                event["subagentRunId"] = json!(subagent_id);
            }
            write_sse(out, &event);
            message.text_open = false;
        }
    }
}

/// What an approval's resume payload looks like, so a client can build its form.
fn approval_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "editedArgs": {"type": "object", "description": "Arguments to run the tool with instead"},
            "reason": {"type": "string", "description": "Why, when not approved"},
        },
        "required": ["approved"],
    })
}
