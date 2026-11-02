//! Tool calls: from the model asking for one to its tool message.
//!
//! The calls of a turn run at the same time (`run_tool_calls`). Each one goes
//! through `before_tool_call` hooks, approval, the tool itself (or the app,
//! the person, a subagent), then `after_tool_call` hooks. A tool that fails
//! gives the model an error result (`failure_result`); only `StopRun` and
//! interrupts end the run.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyox_tools::usage::Usage;
use serde_json::Value;

use crate::agent_loop::StopSignal;
use crate::engine::AgentConfig;
use crate::errors::{HookError, StopRun, ToolFailed, wrap_user_error};
use crate::events::{AgentEvent, ToolFinished};
use crate::hooks::call_python_hook;
use crate::human_input::{
    CallLog, SharedCallLog, approval_required, check_reply, new_request, reply_to, waiting_calls,
};
use crate::json_convert::{json_to_python, python_to_json};
use crate::messages::{Message, Reply, RequestKind, ToolCall};
use crate::python_call::PythonCalls;
use crate::run_context::Context;
use crate::state::apply_state_changes;
use crate::subagents::{question_of, take_subagent_answers};
use crate::telemetry::Span;
use crate::tools::{ToolDefinition, ToolImplementation, ToolResult, error_result, success_result, tool_failed};

/// A tool call to run, and what a resumed run brings for it.
pub struct CallToRun {
    call: ToolCall,
    /// The call's waiting message, when a resumed run continues it.
    earlier: Option<Message>,
    /// Replies for the waiting calls of the subagent the call runs.
    subagent_answers: HashMap<String, Reply>,
}

impl CallToRun {
    /// A call the model just asked for.
    pub fn new(call: ToolCall) -> Self {
        CallToRun {
            call,
            earlier: None,
            subagent_answers: HashMap::new(),
        }
    }
}

/// How one tool call ended: its tool message, the tokens it used (when it
/// ran a subagent), and the error that ends the run, if it raised one.
pub struct CallOutcome {
    message: Message,
    usage: Usage,
    run_ended: Option<PyErr>,
}

/// Runs the calls at the same time and returns how each ended, in the
/// calls' order: the order the model reads the results in.
///
/// Each call sends its `ToolFinished` as soon as it ends, so the app sees a
/// fast call's result while slower calls still run. A call that ends the
/// run (a `StopRun`, say) gets a message too ("The run was stopped: ..."),
/// and the other calls finish first: every call in `error.messages` has a
/// result, so the conversation can be continued.
pub async fn run_tool_calls(run: &ToolRun<'_>, calls: Vec<CallToRun>) -> Vec<CallOutcome> {
    join_all(calls.into_iter().map(|to_run| async move {
        let CallToRun {
            call,
            earlier,
            subagent_answers,
        } = to_run;
        let outcome = match run_tool_call(run, &call, earlier.as_ref(), subagent_answers).await {
            Ok((message, usage)) => CallOutcome {
                message,
                usage,
                run_ended: None,
            },
            Err(error) => {
                let note = format!(
                    "The run was stopped: {}",
                    Python::attach(|py| error.value(py).to_string())
                );
                CallOutcome {
                    message: Message::tool_result(call.id.clone(), call.name.clone(), note, true),
                    usage: Usage::default(),
                    run_ended: Some(error),
                }
            }
        };
        if let Some(finished) = tool_finished(&outcome.message) {
            run.context
                .human_input
                .events
                .send(AgentEvent::ToolFinished(finished))
                .await;
        }
        outcome
    }))
    .await
}

/// The `ToolFinished` event for a tool message, or `None` while the call
/// waits (it has no result yet).
fn tool_finished(message: &Message) -> Option<ToolFinished> {
    let Message::Tool {
        tool_call_id,
        name,
        content,
        is_error,
        metadata,
        requests,
        ..
    } = message
    else {
        return None;
    };
    if message.is_waiting() {
        return None;
    }
    Some(ToolFinished {
        call_id: tool_call_id.clone(),
        name: name.clone(),
        result: content.text(),
        content: content.clone(),
        is_error: *is_error,
        metadata: metadata.clone(),
        from_app: requests.iter().any(|request| request.kind == RequestKind::Frontend),
        subagent_id: None, // set by the event sender
    })
}

/// Adds the calls' tool messages to the history and their tokens to
/// `usage`. Returns the first error that ends the run, if any.
pub fn add_tool_messages(outcomes: Vec<CallOutcome>, messages: &mut Vec<Message>, usage: &mut Usage) -> Option<PyErr> {
    let mut first_error = None;
    for outcome in outcomes {
        usage.add(outcome.usage);
        messages.push(outcome.message);
        if let Some(error) = outcome.run_ended {
            first_error.get_or_insert(error);
        }
    }
    first_error
}

/// The waiting calls that got a reply in `answers`, to run again.
///
/// An answer for `call_0/call_1` is for the call `call_1` of the subagent
/// that the call `call_0` runs: `call_0` runs again, and passes it on.
pub fn answered_calls(history: &[Message], mut answers: HashMap<String, Reply>) -> PyResult<Vec<CallToRun>> {
    let mut answered = Vec::new();
    for (call, waiting_message) in waiting_calls(history) {
        let reply = answers.remove(&call.id);
        let subagent_answers = take_subagent_answers(&mut answers, &call.id);
        if reply.is_none() && subagent_answers.is_empty() {
            continue; // not answered yet: it stays waiting
        }
        let mut message = waiting_message.clone();
        if let Some(reply) = reply {
            // Put the reply on the request the call is waiting on.
            let Message::Tool { requests, .. } = &mut message else {
                unreachable!("waiting calls always have a tool message");
            };
            let Some(request) = requests.iter_mut().find(|request| request.answer.is_none()) else {
                return Err(PyValueError::new_err(format!(
                    "Tool call {:?} waits for its subagent: answer the subagent's requests (see `RunFinished.pending`)",
                    call.id
                )));
            };
            check_reply(request.kind, &reply)?;
            request.answer = Some(reply);
        }
        answered.push(CallToRun {
            call,
            earlier: Some(message),
            subagent_answers,
        });
    }
    if let Some(call_id) = answers.keys().next() {
        return Err(PyValueError::new_err(format!(
            "Tool call {call_id:?} is not waiting for an answer"
        )));
    }
    Ok(answered)
}

/// What running a tool call needs from the run it belongs to.
pub struct ToolRun<'a> {
    pub config: &'a AgentConfig,
    pub tools: &'a [Arc<ToolDefinition>],
    /// The conversation so far: a subagent tool finds earlier conversations in it.
    pub history: &'a [Message],
    pub context: &'a Context,
    pub python_calls: &'a PythonCalls,
    /// Set by `run.stop()`: a subagent this call runs stops too.
    pub stop: &'a StopSignal,
    /// The run's span: each call's span goes under it.
    pub span: &'a Span,
}

/// Runs one tool call and returns its tool message, which is *waiting* when
/// a request got no reply, and the tokens it used (when it ran a subagent).
/// `earlier` is the call's waiting message when a resumed run continues it:
/// its replies and saved steps are reused. `subagent_answers` are replies for
/// the waiting calls of the subagent it runs.
///
/// When the run is traced, the call has its own `execute_tool` span, and its
/// Python code runs with that span as the current one.
async fn run_tool_call(
    run: &ToolRun<'_>,
    call: &ToolCall,
    earlier: Option<&Message>,
    subagent_answers: HashMap<String, Reply>,
) -> PyResult<(Message, Usage)> {
    let tool = run.tools.iter().find(|tool| tool.name == call.name);
    let description = tool.map_or("", |tool| tool.description.as_str());
    let kind = tool.map_or("unknown", |tool| tool.implementation.kind());
    let (span, python_calls) = run.span.start_tool(call, description, kind, run.python_calls).await;
    // The same run, with this call's PythonCalls (`..*run` copies the other fields).
    let run = ToolRun {
        python_calls: &python_calls,
        ..*run
    };
    let outcome = tool_message(&run, call, earlier, subagent_answers, span.clone()).await;
    span.finish_tool(&outcome).await;
    outcome
}

/// The body of `run_tool_call`.
async fn tool_message(
    run: &ToolRun<'_>,
    call: &ToolCall,
    earlier: Option<&Message>,
    subagent_answers: HashMap<String, Reply>,
    span: Span,
) -> PyResult<(Message, Usage)> {
    let call_log = CallLog::new(call.clone(), earlier, span);
    call_log.lock().unwrap().subagent_answers = subagent_answers;
    let result = tool_call_result(run, &call_log).await?;
    // `None` means waiting: no result yet.
    let result = result.unwrap_or_else(|| success_result(String::new(), Value::Null));

    let mut call_log = call_log.lock().unwrap();
    let message = Message::Tool {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        content: result.content,
        is_error: result.is_error,
        metadata: result.metadata,
        requests: call_log.requests.clone(),
        steps: call_log.steps.clone(),
        subagent: std::mem::take(&mut call_log.subagent),
    };
    Ok((message, call_log.subagent_usage))
}

/// The result of one tool call, or `None` when it waits for a person.
///
/// In order: `before_tool_call` hooks, approval (if the tool or a hook asks
/// for it), the tool itself (or the app, for a frontend tool), then
/// `after_tool_call` hooks. A tool hook that fails blocks the call: the model
/// gets the error as the result. A tool that fails goes through
/// `failure_result`. `Err` ends the run (`StopRun`, for one).
async fn tool_call_result(run: &ToolRun<'_>, call_log: &SharedCallLog) -> PyResult<Option<ToolResult>> {
    let ToolRun {
        config,
        tools,
        context,
        python_calls,
        ..
    } = *run;
    let hooks = &config.hooks;
    let model_call = call_log.lock().unwrap().call.clone();
    let mut approval_request = None;
    let mut call = match hooks.before_tool_call(context, model_call.clone(), python_calls).await {
        Ok(call) => call,
        // A hook raising `ApprovalRequired` does not block the call: it asks first.
        Err(error) => match approval_required(&error)? {
            Some(request) => {
                approval_request = Some(request);
                model_call
            }
            None => return blocked_by_hook(error).map(Some),
        },
    };
    let Some(tool) = tools.iter().find(|tool| tool.name == call.name) else {
        let unknown = tool_failed(format!("Unknown tool {:?}", call.name));
        return failure_result(run, None, &call, unknown).await.map(Some);
    };
    call_log.lock().unwrap().call = call.clone();

    if approval_request.is_none() {
        approval_request = match tool.approval_request(call_log, context, python_calls).await {
            Ok(request) => request,
            Err(error) => return failure_result(run, Some(tool), &call, error).await.map(Some),
        };
    }
    if let Some(request) = approval_request {
        match reply_to(call_log, request, &context.human_input, python_calls).await? {
            None => return Ok(None),
            Some(Reply::Deny { message, .. }) => return Ok(Some(error_result(message))),
            Some(Reply::Approve {
                arguments: Some(arguments),
                ..
            }) => {
                call.arguments = arguments;
                call_log.lock().unwrap().call = call.clone();
            }
            Some(_) => {} // approved as is
        }
    }

    let outcome = match &tool.implementation {
        // The app runs a frontend tool, and the person answers `ask_user`: the reply is the result.
        ToolImplementation::Frontend | ToolImplementation::AskUser => {
            let request = match tool.implementation {
                ToolImplementation::Frontend => new_request(RequestKind::Frontend, call.arguments.clone(), Value::Null),
                _ => new_request(RequestKind::Input, question_of(&call), Value::Null),
            };
            match reply_to(call_log, request, &context.human_input, python_calls).await? {
                None => return Ok(None),
                Some(Reply::Answer { value, metadata }) => Ok(success_result(text_for_model(value), metadata)),
                Some(Reply::Deny { message, .. }) => Ok(error_result(message)),
                Some(Reply::Approve { .. }) => unreachable!("check_reply only lets Answer or Deny through"),
            }
        }
        // The subagent's run stops after this turn (see `subagents`).
        ToolImplementation::AskParent => Ok(success_result(
            "Your question was sent. The answer comes in the next message.".to_string(),
            Value::Null,
        )),
        ToolImplementation::Subagent(subagent) => match subagent.run(call_log, run).await? {
            Some(outcome) => outcome,
            None => return Ok(None),
        },
        ToolImplementation::Code(code) => {
            let outcome = tool.run_code(code, &call, call_log, context, python_calls).await;
            // A `ctx.ask` inside the tool may have stopped it, or `on_request` failed in it.
            let denied = {
                let mut call_log = call_log.lock().unwrap();
                if let Some(error) = call_log.on_request_failed.take() {
                    return Err(error);
                }
                if call_log.is_waiting() {
                    return Ok(None);
                }
                call_log.denied.clone()
            };
            match (denied, outcome) {
                (Some(message), _) => Ok(error_result(message)),
                // The tool finished: apply its changes to `ctx.state`.
                (None, Ok(result)) => match apply_state_changes(call_log, context).await {
                    Ok(()) => Ok(result),
                    Err(message) => Err(tool_failed(message)),
                },
                (None, Err(error)) => Err(error),
            }
        }
    };
    let result = match outcome {
        Ok(result) => result,
        Err(error) => failure_result(run, Some(tool), &call, error).await?,
    };

    match hooks.after_tool_call(context, &call, result, python_calls).await {
        Ok(result) => Ok(Some(result)),
        Err(error) => blocked_by_hook(error).map(Some),
    }
}

/// What the model reads when a tool fails. The tool's own `on_error`, then
/// each `on_tool_error` hook, may choose, in that order; the first that
/// returns something wins:
///
/// - `None`: no choice, ask the next one (in the end, the default text);
/// - a `str`: the model reads it, as an error result;
/// - a `ToolResult` dict (`{"content": ..., "is_error": ...}`): the result as it is;
/// - raising ends the run (a `StopRun` as it is, anything else as a `HookError`).
///
/// The default text is the message of a `ToolFailed`, and otherwise the
/// exception's type and message ("ValueError: city not found"); such an
/// unexpected exception is also logged, with its traceback, on the
/// `pyox_agents` logger.
///
/// `StopRun` and non-`Exception`s (`KeyboardInterrupt`, `CancelledError`)
/// are not tool failures: they end the run, untouched.
async fn failure_result(
    run: &ToolRun<'_>,
    tool: Option<&ToolDefinition>,
    call: &ToolCall,
    error: PyErr,
) -> PyResult<ToolResult> {
    if Python::attach(|py| error.is_instance_of::<StopRun>(py) || !error.is_instance_of::<PyException>(py)) {
        return Err(error);
    }
    let tool_on_error = tool.and_then(|tool| tool.on_error.as_ref());
    let handlers = tool_on_error.into_iter().chain(&run.config.hooks.tool_error_handlers);
    for handler in handlers {
        let (handler, call_json, exception) = Python::attach(|py| {
            let call_json = serde_json::to_value(call).expect("tool calls always serialize");
            (handler.clone_ref(py), call_json, error.value(py).clone().unbind())
        });
        let choice = call_python_hook(handler, run.context, run.python_calls, move |py| {
            Ok(vec![
                json_to_python(py, &call_json)?,
                exception.into_bound(py).into_any(),
            ])
        })
        .await
        .map_err(|raised| wrap_user_error::<HookError>(raised, "The on_tool_error hook"))?;
        if let Some(result) = Python::attach(|py| tool_result_from_choice(choice.bind(py)))? {
            return Ok(result);
        }
    }
    Python::attach(|py| {
        if error.is_instance_of::<ToolFailed>(py) {
            return Ok(error_result(error.value(py).to_string()));
        }
        let logger = py.import("logging")?.call_method1("getLogger", ("pyox_agents",))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("exc_info", error.value(py))?;
        logger.call_method("warning", ("Tool %r failed", &call.name), Some(&kwargs))?;
        // `to_string()` gives e.g. "ValueError: city not found".
        Ok(error_result(error.to_string()))
    })
}

/// What an `on_error` / `on_tool_error` function chose: `None` (no choice),
/// a text for the model, or a whole `ToolResult`.
fn tool_result_from_choice(choice: &Bound<'_, PyAny>) -> PyResult<Option<ToolResult>> {
    if choice.is_none() {
        return Ok(None);
    }
    if let Ok(text) = choice.extract::<String>() {
        return Ok(Some(error_result(text)));
    }
    serde_json::from_value(python_to_json(choice)?).map(Some).map_err(|_| {
        HookError::new_err(format!(
            "on_tool_error must return None, a str or a ToolResult, got {}",
            choice.repr().map(|text| text.to_string()).unwrap_or_default()
        ))
    })
}

/// A `before_tool_call` or `after_tool_call` hook raised: the model gets the
/// error as the call's result, unless it is a `StopRun` (or not an
/// `Exception`), which ends the run.
fn blocked_by_hook(error: PyErr) -> PyResult<ToolResult> {
    if Python::attach(|py| error.is_instance_of::<StopRun>(py) || !error.is_instance_of::<PyException>(py)) {
        return Err(error);
    }
    Ok(error_result(message_for_model(error)))
}

/// Text stays as it is; anything else is sent as JSON text.
fn text_for_model(value: Value) -> String {
    match value {
        Value::String(text) => text,
        other => other.to_string(),
    }
}

/// The text the model sees when a tool hook fails. A Rust hook's message is
/// used as is; a Python exception includes its type, like tool errors do.
fn message_for_model(error: PyErr) -> String {
    Python::attach(|py| {
        if error.is_instance_of::<HookError>(py) {
            error.value(py).to_string()
        } else {
            error.to_string()
        }
    })
}
