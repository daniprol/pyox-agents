//! The agent loop: one run of an agent, from its prompt to its `RunFinished`.
//!
//! `Engine` (in `engine.rs`) holds the agent's configuration, and `Run` (in
//! `run.rs`) is what Python reads the run's events from. Tool calls are in
//! `tool_call.rs`.
//!
//! The loop (in `run_agent_loop`):
//!
//! ```text
//! load the conversation from the history store (if the run names one)
//! before_run hooks
//! repeat until the model stops calling tools (at most `max_turns` times):
//!     before_model_request hooks
//!     stream a model response (Chat or Responses API), forwarding text to Python as it arrives
//!         (over WebSocket, the run keeps one connection for all its turns)
//!     after_model_response hooks, then add the assistant message to the history
//!     run the requested tools (concurrently) and add their results;
//!         each call goes through before_tool_call and after_tool_call hooks
//!     stop early if a tool call is waiting for a person, or `run.stop()` was called
//! after_run hooks
//! save the run's new messages to the history store (if any)
//! ```
//!
//! Every step only touches the hooks when the agent has some (see `hooks`).
//!
//! A tool call can wait for a person (an approval, a `ctx.ask` question, a
//! frontend tool; see `human_input`). If no live `on_request` handler
//! replies, the run stops after the tools and `RunFinished.pending` lists the
//! requests. The next run passes `answers=`: before the first model request
//! it runs the waiting calls again with the replies, then carries on.
//!
//! A subagent tool runs this same loop for another agent, inside the tool
//! call (see `subagents`).
//!
//! Every run ends with its history: `RunFinished.messages` and a
//! `stop_reason`, or an error whose `messages` hold the conversation so far.
//! `run.stop()` ends a run at the next safe point: a streaming answer at once
//! (its text so far becomes the assistant message), running tools when they
//! finish. Then `after_run` hooks run and the history store saves, as usual.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures::FutureExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
// `as`: the hooks' request has the same name as the adapters' `ModelRequest`.
use pyox_tools::hooks::ModelRequest as HookRequest;
use pyox_tools::usage::Usage;
use tokio::sync::watch;

use crate::engine::{AgentConfig, Unanswered};
use crate::errors::{AgentError, HookError, MaxTurnsExceededError, PendingRequestsError, with_messages, with_usage};
use crate::events::{AgentEvent, RunFinished, ToolStarted};
use crate::human_input::{HumanInput, pending_requests, unanswered_note, waiting_calls};
use crate::messages::{Content, Message, Reply};
use crate::model_api::{ModelRequest, StopReason, last_response_id, stream_model_turn};
use crate::python_call::PythonCalls;
use crate::run_context::Context;
use crate::state::RunState;
use crate::subagents::asked_parent;
use crate::telemetry::{ModelRequestRecord, Span, Telemetry, now_ns};
use crate::text_chunks::{TextChunking, TextOutput};
use crate::tool_call::{CallToRun, ToolRun, add_tool_messages, answered_calls, run_tool_calls};
use crate::tools::ToolDefinition;
use crate::ws_pool::RunSocket;

/// What a run needs to start. Kept aside until the first event is requested.
pub struct RunInputs {
    pub config: Arc<AgentConfig>,
    pub text_chunks: TextChunking,
    /// History given directly with `execute(history=...)`.
    pub history: Vec<Message>,
    pub prompt: Option<Content>,
    /// Set when the history is loaded from (and saved to) the history store.
    pub conversation_id: Option<String>,
    /// The stored response this run continues, from `execute(previous_response_id=...)`.
    pub previous_response_id: Option<String>,
    /// Given to tools and hooks as `ctx.deps`.
    pub deps: Arc<Py<PyAny>>,
    /// Replies to the tool calls the last run left waiting, by call id.
    pub answers: HashMap<String, Reply>,
    /// The live `on_request` handler, and where the run's events go.
    pub human_input: Arc<HumanInput>,
    /// Tools added for this run only (`execute(tools=..., frontend_tools=...)`),
    /// or what a subagent gets from its tool.
    pub extra_tools: Vec<Arc<ToolDefinition>>,
    pub state: RunState,
    /// Set by `run.stop()`; a subagent shares its caller's.
    pub stop: StopSignal,
    /// Where the run's span goes, if the run is traced.
    pub telemetry: Telemetry,
}

/// `true` once `run.stop()` was called. A `watch` channel holds one value:
/// one side sets it, the other reads it or waits until it changes (like an
/// `asyncio.Event` that can be shared between threads).
pub type StopSignal = watch::Receiver<bool>;

/// Waits until `run.stop()` is called.
async fn stop_called(stop: &StopSignal) {
    let mut stop = stop.clone();
    if stop.wait_for(|stopped| *stopped).await.is_err() {
        // The `Run` is gone, so nobody can stop it: wait forever (the
        // dropped run aborts this task anyway).
        std::future::pending::<()>().await;
    }
}

/// Runs the agent loop and makes sure any failure reaches Python as an exception.
pub async fn run_in_background(inputs: RunInputs, python_calls: PythonCalls) {
    let events = inputs.human_input.events.clone();
    let mut usage = Usage::default();
    let mut messages = Vec::new();
    let loop_result = AssertUnwindSafe(run_agent_loop(inputs, python_calls, &mut usage, &mut messages))
        .catch_unwind()
        .await;

    let error = match loop_result {
        Ok(Ok(finished)) => return events.send(AgentEvent::RunFinished(finished)).await,
        Ok(Err(python_error)) => with_messages(with_usage(python_error, usage), &messages),
        // A panic is a bug in this library: report it as a normal error
        // (Python's `PanicException` is not an `Exception`, so `except Exception` would miss it).
        Err(panic) => AgentError::new_err(format!(
            "pyox-agents crashed while running the agent (this is a bug, please report it): {}",
            panic_message(panic.as_ref())
        )),
    };
    events.send_error(error).await;
}

/// The text a panic was raised with (`panic!("...")`), if it has one.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    // A panic carries any value; it is almost always text, in one of two Rust string types.
    match (panic.downcast_ref::<&str>(), panic.downcast_ref::<String>()) {
        (Some(text), _) => text.to_string(),
        (_, Some(text)) => text.clone(),
        _ => "no message".to_string(),
    }
}

/// Runs an agent until it answers (or its tool calls wait for a person), and
/// returns the `RunFinished` event (the main run sends it; a subagent's tool
/// reads it).
///
/// `usage` and `messages` (the conversation) are kept up to date as the run
/// goes, so the caller has them even when the run fails. On success they
/// move into the `RunFinished`.
///
/// When the run is traced, everything happens inside its `invoke_agent`
/// span, and its Python calls see that span as the current one.
pub async fn run_agent_loop(
    inputs: RunInputs,
    python_calls: PythonCalls,
    usage: &mut Usage,
    messages: &mut Vec<Message>,
) -> PyResult<RunFinished> {
    let config = inputs.config.clone();
    let (run_span, python_calls) = inputs
        .telemetry
        .start_run(
            config.name.as_deref(),
            &config.model,
            &config.client.base_url,
            inputs.conversation_id.as_deref(),
            &python_calls,
        )
        .await;
    let outcome = agent_loop(inputs, python_calls, &run_span, usage, messages).await;
    run_span.finish_run(&outcome, *usage).await;
    outcome
}

/// The body of `run_agent_loop`.
async fn agent_loop(
    inputs: RunInputs,
    python_calls: PythonCalls,
    run_span: &Span,
    usage: &mut Usage,
    messages: &mut Vec<Message>,
) -> PyResult<RunFinished> {
    let RunInputs {
        config,
        history,
        prompt,
        conversation_id,
        previous_response_id: run_start_response_id,
        text_chunks,
        deps,
        answers,
        human_input,
        extra_tools,
        state,
        stop,
        telemetry: _, // used by `run_agent_loop`
    } = inputs;
    // The tools of this run: the agent's, plus the ones added for this run.
    let tools: Vec<Arc<ToolDefinition>> = config.tools.iter().cloned().chain(extra_tools).collect();
    let events = human_input.events.clone();
    // The store and the conversation, when this run loads and saves one.
    let stored_conversation = match &conversation_id {
        Some(conversation_id) => config.history_store.as_ref().map(|store| (store, conversation_id)),
        None => None,
    };

    *messages = match stored_conversation {
        Some((store, conversation_id)) => store.load(conversation_id, &python_calls).await?,
        None => history,
    };
    // Everything from here on is new in this run (and is what gets saved).
    let new_messages_start = messages.len();

    // Tool calls the last run left waiting need answers, or are closed by a new prompt.
    let waiting_count = waiting_calls(messages).len();
    if waiting_count > 0 && answers.is_empty() {
        if prompt.is_none() {
            return Err(PyValueError::new_err(format!(
                "{waiting_count} tool call(s) are waiting for a person: pass `answers={{call_id: ...}}`"
            )));
        }
        let closed_calls = close_waiting_calls(messages, config.unanswered)?;
        messages.extend(closed_calls);
    }
    if let Some(prompt) = prompt {
        messages.push(Message::User { content: prompt });
    }
    if messages.is_empty() {
        return Err(PyValueError::new_err(format!(
            "Conversation {:?} has no messages yet: give a prompt",
            conversation_id.unwrap_or_default()
        )));
    }
    let mut text_output = TextOutput::new(text_chunks, events.clone());
    let hooks = &config.hooks;
    // What tools and hooks see as `ctx`; updated as the run goes.
    let mut context = Context {
        run: pyox_tools::hooks::RunContext {
            conversation_id: conversation_id.clone(),
            ..Default::default()
        },
        deps,
        human_input,
        state,
        text_chunks,
    };
    hooks.before_run(&context, &python_calls).await?;
    // The WebSocket connection of this run (`Client(transport="websocket")`).
    // It is closed when this function returns, unless given back to the pool.
    let mut run_socket = RunSocket::default();

    // Finishes the run: `after_run` hooks, then saving the new messages.
    // (A closure: like a small local function that can use the variables around it.)
    let finish = async |messages: &mut Vec<Message>,
                        text: String,
                        turns: u32,
                        response_id: Option<String>,
                        stop_reason: StopReason,
                        context: &Context|
           -> PyResult<RunFinished> {
        let pending = pending_requests(messages);
        let stop_reason = if pending.is_empty() {
            stop_reason
        } else {
            StopReason::Waiting
        };
        // Only a finished answer is parsed: one cut short or refused, or a
        // subagent stopping to ask its parent (a tool call), gives `None`.
        let answered = stop_reason == StopReason::End
            && matches!(messages.last(), Some(Message::Assistant { tool_calls, .. }) if tool_calls.is_empty());
        let output = match &config.output {
            Some(output) if answered => Some(Arc::new(output.parse(&text)?)),
            _ => None,
        };
        let finished = RunFinished {
            text,
            turns,
            usage: context.run.usage,
            response_id,
            // `mem::take` moves the messages out, leaving an empty list behind.
            messages: std::mem::take(messages),
            new_messages_start,
            stop_reason,
            output,
            pending,
            state: context.state.current(),
        };
        let saved = async {
            hooks.after_run(context, &finished, &python_calls).await?;
            if let Some((store, conversation_id)) = stored_conversation {
                store
                    .save(conversation_id, finished.added_messages(), &python_calls)
                    .await?;
            }
            Ok(())
        };
        match saved.await {
            Ok(()) => Ok(finished),
            Err(error) => {
                // Give the messages back, so the error carries the finished
                // conversation (`error.messages`): the work is not lost.
                *messages = finished.messages;
                Err(error)
            }
        }
    };

    if !answers.is_empty() {
        let resumed = {
            let run = ToolRun {
                config: &config,
                tools: &tools,
                history: messages,
                context: &context,
                python_calls: &python_calls,
                stop: &stop,
                span: run_span,
            };
            run_tool_calls(&run, answered_calls(messages, answers)?).await
        };
        let run_ended = add_tool_messages(resumed, messages, usage);
        context.run.usage = *usage;
        if let Some(error) = run_ended {
            return Err(error);
        }
        if !pending_requests(messages).is_empty() {
            return finish(messages, String::new(), 0, None, StopReason::Waiting, &context).await;
        }
    }

    for turn in 1..=config.max_turns {
        // `run.stop()` came while the tools ran (or before the run started): send nothing more.
        if *stop.borrow() {
            return finish(messages, String::new(), turn - 1, None, StopReason::Stopped, &context).await;
        }
        context.run.turn = turn;
        // Hooks may change this one request (instructions, messages, tools,
        // model parameters); the history stays as it is.
        let hooked = hooks
            .before_model_request(&context, || hook_request(&config, &tools, messages), &python_calls)
            .await?;
        let request_tools = match &hooked {
            Some(hooked) => tools_named(&tools, &hooked.tools)?,
            None => tools.clone(), // cheap: a list of shared pointers
        };
        // The request can continue from the newest response and send only the
        // messages after it, when OpenAI still has that response (the adapter
        // decides). If hooks changed the messages, OpenAI's copy no longer
        // matches them: send them all (on top of the response the run started
        // from, if any).
        let messages_changed = hooked.as_ref().is_some_and(|hooked| hooked.messages != **messages);
        let previous_response_id = if messages_changed {
            run_start_response_id.as_deref()
        } else {
            last_response_id(messages).or(run_start_response_id.as_deref())
        };
        let output_schema = config.output.as_ref().map(|output| &output.schema);
        let request = match &hooked {
            Some(hooked) => ModelRequest {
                model: &config.model,
                instructions: &hooked.instructions,
                messages: &hooked.messages,
                tools: &request_tools,
                model_parameters: &hooked.model_parameters,
                previous_response_id,
                output_schema,
            },
            None => ModelRequest {
                model: &config.model,
                instructions: &config.instructions,
                messages,
                tools: &request_tools,
                model_parameters: &config.model_parameters,
                previous_response_id,
                output_schema,
            },
        };
        // `select!` runs both at once and keeps the first to finish, dropping
        // the other (like `asyncio.wait(..., return_when=FIRST_COMPLETED)`
        // followed by cancelling the rest). Dropping the stream closes it.
        let started_at = now_ns();
        let streamed = tokio::select! {
            assistant_turn = stream_model_turn(&config.client, config.api, &request, &mut run_socket, &mut text_output) => Some(assistant_turn),
            () = stop_called(&stop) => None,
        };
        let record = ModelRequestRecord {
            turn,
            started_at,
            ended_at: now_ns(),
            request: &request,
            outcome: streamed.as_ref(),
        };
        run_span.record_model_request(record).await;
        // `transpose` turns "maybe a result" into "a result that is maybe
        // there", so `?` can raise the request's error.
        let streamed = streamed.transpose()?;
        // The message ended: send any text still grouped.
        let streamed_text = text_output.finish().await;
        let Some(assistant_turn) = streamed else {
            // Stopped mid-answer: keep the text so far (its tool calls were
            // not complete). Its tokens are not known: the usage arrives last.
            if !streamed_text.is_empty() {
                messages.push(Message::Assistant {
                    content: streamed_text.clone(),
                    tool_calls: Vec::new(),
                    reasoning: Vec::new(),
                    response_id: None,
                    model: Some(config.model.clone()),
                });
            }
            return finish(messages, streamed_text, turn, None, StopReason::Stopped, &context).await;
        };
        usage.add(assistant_turn.usage);
        context.run.usage = *usage;

        // An answer cut short may end in a half-written tool call: never run one.
        let tool_calls = match assistant_turn.stop_reason {
            StopReason::End => assistant_turn.tool_calls,
            _ => Vec::new(),
        };
        let assistant_message = Message::Assistant {
            content: assistant_turn.text,
            tool_calls,
            reasoning: assistant_turn.reasoning,
            response_id: assistant_turn.response_id,
            model: Some(config.model.clone()),
        };
        let assistant_message = hooks
            .after_model_response(&context, assistant_message, &python_calls)
            .await?;
        let Message::Assistant {
            content,
            tool_calls,
            response_id,
            ..
        } = &assistant_message
        else {
            unreachable!("after_model_response always gives an assistant message");
        };
        let (text, tool_calls, response_id) = (content.clone(), tool_calls.clone(), response_id.clone());
        messages.push(assistant_message);

        if !tool_calls.is_empty() {
            for call in &tool_calls {
                let started = ToolStarted {
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    subagent_id: None, // set by the event sender
                };
                events.send(AgentEvent::ToolStarted(started)).await;
            }

            // Run all requested tools at the same time and wait for all of them.
            let outcomes = {
                let run = ToolRun {
                    config: &config,
                    tools: &tools,
                    history: messages,
                    context: &context,
                    python_calls: &python_calls,
                    stop: &stop,
                    span: run_span,
                };
                run_tool_calls(&run, tool_calls.iter().cloned().map(CallToRun::new).collect()).await
            };
            let run_ended = add_tool_messages(outcomes, messages, usage);
            context.run.usage = *usage;
            if let Some(error) = run_ended {
                return Err(error);
            }
        }

        let waiting = !pending_requests(messages).is_empty();
        // A subagent stops when it asks its calling agent something (see `subagents`).
        if tool_calls.is_empty() || waiting || asked_parent(&tools, &tool_calls) {
            // The last answer arrived in full, so the connection is clean: the next run may use it.
            run_socket.give_back(&config.client);
            let stop_reason = assistant_turn.stop_reason;
            return finish(messages, text, turn, response_id, stop_reason, &context).await;
        }
    }

    Err(MaxTurnsExceededError::new_err(format!(
        "Stopped after {} turns",
        config.max_turns
    )))
}

/// The request `before_model_request` hooks see: what the agent would send.
fn hook_request(config: &AgentConfig, tools: &[Arc<ToolDefinition>], messages: &[Message]) -> HookRequest {
    HookRequest {
        instructions: config.instructions.clone(),
        messages: messages.to_vec(),
        tools: tools.iter().map(|tool| tool.name.clone()).collect(),
        model_parameters: config.model_parameters.clone(),
    }
}

/// The run's tools with these names, in this order (a hook chose them).
fn tools_named(tools: &[Arc<ToolDefinition>], names: &[String]) -> PyResult<Vec<Arc<ToolDefinition>>> {
    names
        .iter()
        .map(|name| {
            let tool = tools.iter().find(|tool| tool.name == *name);
            tool.cloned().ok_or_else(|| {
                HookError::new_err(format!(
                    "before_model_request offered {name:?}, which is not one of the run's tools"
                ))
            })
        })
        .collect()
}

/// For a new prompt while calls wait: one closing tool message per waiting
/// call (so the model sees every call answered), or `PendingRequestsError`.
fn close_waiting_calls(messages: &[Message], unanswered: Unanswered) -> PyResult<Vec<Message>> {
    let waiting = waiting_calls(messages);
    if unanswered == Unanswered::Raise {
        let call_names: Vec<String> = waiting
            .iter()
            .map(|(call, _)| format!("{} ({})", call.id, call.name))
            .collect();
        return Err(PendingRequestsError::new_err(format!(
            "Tool calls are waiting for answers: {}. Pass `answers=` before a new prompt.",
            call_names.join(", ")
        )));
    }
    Ok(waiting
        .iter()
        .map(|(call, message)| Message::tool_result(call.id.clone(), call.name.clone(), unanswered_note(message), true))
        .collect())
}
