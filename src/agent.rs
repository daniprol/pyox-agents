//! The agent loop.
//!
//! - `Engine` holds the agent's configuration. It never changes after creation,
//!   so one engine can serve many runs at the same time.
//! - `Run` (in `run.rs`) is one execution: Python reads its events from it.
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

use crate::python_call::PythonCalls;
use futures::FutureExt;
use futures::future::join_all;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
// `as`: the hooks' request has the same name as the adapters' `ModelRequest`.
use pyox_tools::hooks::ModelRequest as HookRequest;
use pyox_tools::usage::Usage;
use serde_json::{Map, Value};
use tokio::sync::mpsc::channel;
use tokio::sync::watch;

use crate::client::{Client, ClientSettings};
use crate::errors::{
    AgentError, HookError, InvalidResponseError, MaxTurnsExceededError, PendingRequestsError, StopRun, with_messages,
    with_usage, wrap_user_error,
};
use crate::events::{AgentEvent, EventSender, RunFinished, ToolFinished, ToolStarted};
use crate::history_store::{HistoryStore, messages_from_python};
use crate::hooks::{AgentHooks, call_python_hook};
use crate::human_input::{
    CallLog, HumanInput, SharedCallLog, approval_required, check_reply, new_request, pending_requests,
    reply_from_python, reply_to, unanswered_note, waiting_calls,
};
use crate::json_convert::{json_to_python, python_to_json};
use crate::messages::{Content, Message, Reply, RequestKind, ToolCall, content_from_python};
use crate::model_api::{Api, ModelRequest, StopReason, last_response_id, stream_model_turn};
use crate::run::new_run;
use crate::run_context::Context;
use crate::state::{RunState, apply_state_changes};
use crate::subagents::{asked_parent, question_of, take_subagent_answers};
use crate::telemetry::{ModelRequestRecord, Span, Telemetry, now_ns};
use crate::text_chunks::{TextChunking, TextOutput};
use crate::tools::{ToolDefinition, ToolImplementation, ToolResult, error_result, success_result, tool_failed};
use crate::ws_pool::RunSocket;

/// How many events can wait in the queue before the agent loop pauses.
/// A limit keeps memory bounded if Python reads events slowly.
const EVENT_QUEUE_CAPACITY: usize = 64;

/// The configuration shared (read-only) by every run of an agent.
pub struct AgentConfig {
    /// `Agent(name=...)`: names the run's span (see `telemetry`).
    name: Option<String>,
    client: ClientSettings,
    /// Chat Completions or Responses, with its settings.
    api: Api,
    model: String,
    /// The system prompt (`Agent(instructions=...)`).
    instructions: String,
    /// `Arc` (a shared pointer) so each run can list them with its own frontend tools.
    pub tools: Vec<Arc<ToolDefinition>>,
    model_parameters: Map<String, Value>,
    max_turns: u32,
    /// How streamed text is grouped into `TextDelta` events (can be changed per run).
    text_chunks: TextChunking,
    history_store: Option<HistoryStore>,
    hooks: AgentHooks,
    unanswered: Unanswered,
    output: Option<StructuredOutput>,
}

/// `Agent(output_type=...)`: the answer is JSON matching a schema, turned into a Python value.
pub struct StructuredOutput {
    /// `{"name": ..., "schema": ..., "strict": true}`, sent with every request.
    schema: Value,
    /// A Python function: the answer's text in, the value out (made by `Agent`).
    parse: Py<PyAny>,
}

impl StructuredOutput {
    /// The answer as a Python value. The model wrote something else than the
    /// schema asks for (rare in strict mode): `InvalidResponseError`.
    fn parse(&self, text: &str) -> PyResult<Py<PyAny>> {
        // Quick work, so it runs right here instead of on a blocking-work thread.
        Python::attach(|py| {
            self.parse.call1(py, (text,)).map_err(|error| {
                let invalid = InvalidResponseError::new_err(format!(
                    "The answer does not match `output_type`: {}",
                    error.value(py)
                ));
                invalid.set_cause(py, Some(error)); // like `raise ... from error`
                invalid
            })
        })
    }
}

/// What to do with tool calls still waiting for a person when a new prompt
/// comes instead of answers (`Agent(unanswered=...)`).
#[derive(Clone, Copy, PartialEq)]
enum Unanswered {
    /// Give each call a result saying the user did not answer, then go on.
    Close,
    /// Raise `PendingRequestsError`.
    Raise,
}

/// Python: created by `pyox_agents.Agent`; most users never touch it directly.
#[pyclass(frozen, module = "pyox_agents")]
pub struct Engine {
    pub config: Arc<AgentConfig>,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (
        *, client, model, name = None, instructions = String::new(), tools = Vec::new(), model_parameters = None, max_turns = 20,
        text_chunks = None, history_store = None, hooks = Vec::new(),
        api = "chat", store = false, encrypted_reasoning = true, unanswered = "close",
        output_schema = None, output_parser = None,
    ))]
    #[allow(clippy::too_many_arguments)] // mirrors the keyword arguments of `Agent(...)`
    fn new(
        client: PyRef<'_, Client>,
        model: String,
        name: Option<String>,
        instructions: String,
        tools: Vec<Bound<'_, PyAny>>,
        model_parameters: Option<Bound<'_, PyDict>>,
        max_turns: u32,
        text_chunks: Option<Bound<'_, PyAny>>,
        history_store: Option<Bound<'_, PyAny>>,
        hooks: Vec<Bound<'_, PyAny>>,
        api: &str,
        store: bool,
        encrypted_reasoning: bool,
        unanswered: &str,
        output_schema: Option<Bound<'_, PyDict>>,
        output_parser: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let api = Api::from_python(api, store, encrypted_reasoning)?;
        let tools = tools
            .iter()
            .map(|tool| ToolDefinition::from_python(tool).map(Arc::new))
            .collect::<PyResult<Vec<_>>>()?;
        for (position, tool) in tools.iter().enumerate() {
            if tools[..position].iter().any(|earlier| earlier.name == tool.name) {
                return Err(PyValueError::new_err(format!("Two tools are named {:?}", tool.name)));
            }
        }

        let unanswered = match unanswered {
            "close" => Unanswered::Close,
            "raise" => Unanswered::Raise,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown unanswered={other:?}: use \"close\" or \"raise\""
                )));
            }
        };

        let model_parameters = match model_parameters {
            Some(parameters) => match python_to_json(parameters.as_any())? {
                Value::Object(fields) => fields,
                _ => unreachable!("a dict always converts to a JSON object"),
            },
            None => Map::new(),
        };

        let output = match (output_schema, output_parser) {
            (Some(schema), Some(parse)) => Some(StructuredOutput {
                schema: python_to_json(schema.as_any())?,
                parse,
            }),
            (None, None) => None,
            _ => return Err(PyValueError::new_err("Give both `output_schema` and `output_parser`")),
        };

        Ok(Engine {
            config: Arc::new(AgentConfig {
                name,
                client: client.settings.clone(),
                api,
                model,
                instructions,
                tools,
                model_parameters,
                max_turns,
                text_chunks: match text_chunks {
                    Some(value) => TextChunking::from_python(&value)?,
                    None => TextChunking::Delta,
                },
                history_store: history_store.as_ref().map(HistoryStore::from_python).transpose()?,
                hooks: AgentHooks::from_python(&hooks)?,
                unanswered,
                output,
            }),
        })
    }

    /// Starts a run. Nothing is sent until Python starts iterating over it.
    ///
    /// The history comes from `history` (message dicts or JSON text), or is
    /// loaded from the agent's history store when `conversation_id` is given.
    /// `deps` is any object, given to tools and hooks as `ctx.deps`.
    /// `previous_response_id` continues a conversation OpenAI stored (Responses API with `store`).
    /// `answers` replies to the tool calls the last run left waiting (`{call_id: reply}`);
    /// `on_request` replies live, while the run waits (see `human_input`).
    /// `tools` adds tools for this run only (the same kinds as `Agent(tools=...)`).
    /// `frontend_tools` adds tools the app runs, for this run only (dicts with
    /// `name`, `description` and `parameters`, as AG-UI clients send them).
    /// `state` is the run's state, shared with the app (see `state`).
    /// `telemetry` is the recorder that traces this run (see `telemetry`), or `None`.
    #[pyo3(signature = (
        prompt = None, *, history = None, conversation_id = None, text_chunks = None, deps = None,
        previous_response_id = None, answers = None, on_request = None, tools = None, frontend_tools = None,
        state = None, telemetry = None, sync = false,
    ))]
    #[allow(clippy::too_many_arguments)] // mirrors the keyword arguments of `Agent.execute(...)`
    fn execute(
        &self,
        py: Python<'_>,
        prompt: Option<Bound<'_, PyAny>>,
        history: Option<Bound<'_, PyAny>>,
        conversation_id: Option<String>,
        text_chunks: Option<Bound<'_, PyAny>>,
        deps: Option<Py<PyAny>>,
        previous_response_id: Option<String>,
        answers: Option<Bound<'_, PyDict>>,
        on_request: Option<Py<PyAny>>,
        tools: Option<Vec<Bound<'_, PyAny>>>,
        frontend_tools: Option<Vec<Bound<'_, PyAny>>>,
        state: Option<Bound<'_, PyAny>>,
        telemetry: Option<Py<PyAny>>,
        sync: bool,
    ) -> PyResult<Py<PyAny>> {
        let text_chunks = match text_chunks {
            Some(value) => TextChunking::from_python(&value)?,
            None => self.config.text_chunks,
        };
        if conversation_id.is_some() {
            if history.is_some() {
                return Err(PyValueError::new_err("Give `history` or `conversation_id`, not both"));
            }
            if self.config.history_store.is_none() {
                return Err(PyValueError::new_err(
                    "`conversation_id` needs an agent with a `history_store`",
                ));
            }
        }
        if previous_response_id.is_some() {
            if !self.config.api.stores_responses() {
                return Err(PyValueError::new_err(
                    "`previous_response_id` needs an agent with `api=\"responses\"` and `store=True`",
                ));
            }
            // The stored response already holds the conversation.
            if history.is_some() || conversation_id.is_some() {
                return Err(PyValueError::new_err(
                    "Give `previous_response_id` or a history (`history`, `conversation_id`), not both",
                ));
            }
            if prompt.is_none() {
                return Err(PyValueError::new_err("`previous_response_id` needs a prompt"));
            }
        }
        let mut replies = HashMap::new();
        for (call_id, reply) in answers.iter().flat_map(|answers| answers.iter()) {
            let reply = reply_from_python(&reply)?
                .ok_or_else(|| PyValueError::new_err("An answer cannot be None: use Answer, Approve or Deny"))?;
            replies.insert(call_id.extract::<String>()?, reply);
        }
        if !replies.is_empty() {
            if prompt.is_some() {
                return Err(PyValueError::new_err(
                    "Give `answers` or a prompt, not both: answer the waiting tool calls first",
                ));
            }
            if previous_response_id.is_some() {
                return Err(PyValueError::new_err(
                    "`answers` needs the history of the stopped run (`history` or `conversation_id`), \
                     not `previous_response_id`",
                ));
            }
        }
        // Tools for this run only: the app's own, then its frontend tools.
        let mut run_tools: Vec<Arc<ToolDefinition>> = Vec::new();
        for tool in tools.iter().flatten() {
            run_tools.push(Arc::new(ToolDefinition::from_python(tool)?));
        }
        for tool in frontend_tools.iter().flatten() {
            run_tools.push(Arc::new(ToolDefinition::frontend_from_python(tool)?));
        }
        for (index, tool) in run_tools.iter().enumerate() {
            let taken = self.config.tools.iter().chain(&run_tools[..index]);
            if taken.map(|other| &other.name).any(|name| *name == tool.name) {
                return Err(PyValueError::new_err(format!(
                    "Two tools are named {:?}: tools given for a run need names of their own",
                    tool.name
                )));
            }
        }
        let prompt = prompt.as_ref().map(content_from_python).transpose()?;
        let history = match history {
            Some(history) => messages_from_python(&history)?,
            None => Vec::new(),
        };
        // A run with a conversation_id is checked after loading, in the loop.
        if prompt.is_none() && history.is_empty() && conversation_id.is_none() {
            return Err(PyValueError::new_err("Give a prompt, a history, or both"));
        }

        let state = match state {
            Some(state) => python_to_json(&state)?,
            None => Value::Null,
        };

        let (event_sender, event_receiver) = channel(EVENT_QUEUE_CAPACITY);
        // `run.stop()` sets this to `true` (see `StopSignal`).
        let (stop_sender, stop) = watch::channel(false);
        let inputs = RunInputs {
            config: self.config.clone(),
            history,
            prompt,
            conversation_id,
            previous_response_id,
            text_chunks,
            deps: Arc::new(deps.unwrap_or_else(|| py.None())),
            answers: replies,
            human_input: Arc::new(HumanInput {
                on_request,
                events: EventSender::new(event_sender),
            }),
            extra_tools: run_tools,
            state: RunState::new(state),
            stop,
            telemetry: match telemetry {
                Some(recorder) => Telemetry::Recorder(Arc::new(recorder)),
                None => Telemetry::Off,
            },
        };
        new_run(py, inputs, event_receiver, stop_sender, sync)
    }
}

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

/// A tool call to run, and what a resumed run brings for it.
struct CallToRun {
    call: ToolCall,
    /// The call's waiting message, when a resumed run continues it.
    earlier: Option<Message>,
    /// Replies for the waiting calls of the subagent the call runs.
    subagent_answers: HashMap<String, Reply>,
}

impl CallToRun {
    /// A call the model just asked for.
    fn new(call: ToolCall) -> Self {
        CallToRun {
            call,
            earlier: None,
            subagent_answers: HashMap::new(),
        }
    }
}

/// How one tool call ended: its tool message, the tokens it used (when it
/// ran a subagent), and the error that ends the run, if it raised one.
struct CallOutcome {
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
async fn run_tool_calls(run: &ToolRun<'_>, calls: Vec<CallToRun>) -> Vec<CallOutcome> {
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
            run.context.human_input.events.send(AgentEvent::ToolFinished(finished)).await;
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
fn add_tool_messages(outcomes: Vec<CallOutcome>, messages: &mut Vec<Message>, usage: &mut Usage) -> Option<PyErr> {
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
fn answered_calls(history: &[Message], mut answers: HashMap<String, Reply>) -> PyResult<Vec<CallToRun>> {
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
        _ => {
            let outcome = tool.run_with_timeout(&call, call_log, context, python_calls).await;
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
        let tool_failed = py.import("pyox_agents.tools")?.getattr("ToolFailed")?;
        if error.value(py).is_instance(&tool_failed)? {
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
