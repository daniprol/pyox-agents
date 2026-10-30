//! A run as an AG-UI event stream (CopilotKit and other AG-UI clients).
//! Python: `StreamingResponse(run.ag_ui(thread_id, run_id), headers=ag_ui.HEADERS)`.
//!
//! Each run event becomes one or more AG-UI events, each sent as `data: {json}`:
//!
//! | Run event        | AG-UI events                                                         |
//! |------------------|----------------------------------------------------------------------|
//! | (stream starts)  | `RUN_STARTED`                                                        |
//! | `TextDelta`      | `TEXT_MESSAGE_START` (once per message), `TEXT_MESSAGE_CONTENT`      |
//! | `ToolStarted`    | `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`                 |
//! | `ToolFinished`   | `TOOL_CALL_RESULT`                                                   |
//! | `InputRequested` | nothing yet: listed in `RUN_FINISHED`                                |
//! | `RunFinished`    | `RUN_FINISHED`, with an `interrupt` outcome when approvals or questions wait |
//! | a pyox-agents error | `RUN_ERROR`                                                       |
//!
//! Waiting calls, as AG-UI 1.0 expects them:
//! - an approval or a `ctx.ask` question is an *interrupt* whose id is the
//!   tool call id. The client resumes with `resume: [{interruptId, status, payload}]`;
//!   an approval's payload is `{approved, editedArgs?, reason?}`.
//! - a frontend tool is a *pending tool call*: the client runs it and sends
//!   the result as a tool message in the next run.

use serde_json::{Value, json};

use crate::events::AgentEvent;
use crate::messages::RequestKind;
use crate::run::{new_id, write_sse};

pub struct AgUiEncoder {
    thread_id: String,
    run_id: String,
    /// The assistant message of the current model turn (text and tool calls belong to it).
    message_id: Option<String>,
    /// Its text was started (`TEXT_MESSAGE_START`) and not ended yet.
    text_open: bool,
    /// Tool results came after the turn's output: the next output is a new message.
    after_tools: bool,
    /// Approvals and questions waiting, for the `RUN_FINISHED` outcome.
    interrupts: Vec<Value>,
    /// Frontend tool calls waiting for the client.
    pending_tool_call_ids: Vec<String>,
}

impl AgUiEncoder {
    pub fn new(thread_id: String, run_id: String) -> Self {
        AgUiEncoder {
            thread_id,
            run_id,
            message_id: None,
            text_open: false,
            after_tools: false,
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
        match event {
            AgentEvent::TextDelta(delta) => {
                if delta.text.is_empty() {
                    return;
                }
                let message_id = self.current_message_id(out);
                if !self.text_open {
                    write_sse(
                        out,
                        &json!({"type": "TEXT_MESSAGE_START", "messageId": message_id, "role": "assistant"}),
                    );
                    self.text_open = true;
                }
                write_sse(
                    out,
                    &json!({"type": "TEXT_MESSAGE_CONTENT", "messageId": message_id, "delta": delta.text}),
                );
            }
            AgentEvent::ToolStarted(call) => {
                let message_id = self.current_message_id(out);
                self.end_text(out);
                write_sse(
                    out,
                    &json!({
                        "type": "TOOL_CALL_START",
                        "toolCallId": call.call_id,
                        "toolCallName": call.name,
                        "parentMessageId": message_id,
                    }),
                );
                write_sse(
                    out,
                    &json!({"type": "TOOL_CALL_ARGS", "toolCallId": call.call_id, "delta": call.arguments.to_string()}),
                );
                write_sse(out, &json!({"type": "TOOL_CALL_END", "toolCallId": call.call_id}));
            }
            AgentEvent::ToolFinished(result) => {
                self.end_text(out);
                self.after_tools = true;
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
                write_sse(out, &event);
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
                    self.interrupts.push(interrupt);
                }
            },
            AgentEvent::RunFinished(_) => {
                self.end_text(out);
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

    /// The id of the current turn's assistant message; a new one after tool results.
    fn current_message_id(&mut self, out: &mut String) -> String {
        if self.after_tools {
            self.end_text(out);
            self.message_id = None;
            self.after_tools = false;
        }
        self.message_id.get_or_insert_with(|| new_id("msg")).clone()
    }

    /// AG-UI clients reject a run that finishes with a text message still open.
    fn end_text(&mut self, out: &mut String) {
        if self.text_open
            && let Some(message_id) = &self.message_id
        {
            write_sse(out, &json!({"type": "TEXT_MESSAGE_END", "messageId": message_id}));
            self.text_open = false;
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
