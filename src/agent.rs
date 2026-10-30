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
//!     stop early if a tool call is waiting for a person
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

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::join_all;
use pyo3::exceptions::PyValueError;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_async_runtimes::TaskLocals;
use pyox_tools::usage::Usage;
use serde_json::{Map, Value};
use tokio::sync::mpsc::{Sender, channel};

use crate::client::{Client, ClientSettings};
use crate::errors::{HookError, MaxTurnsExceededError, PendingRequestsError, with_usage};
use crate::events::{AgentEvent, RunFinished, ToolFinished, ToolStarted};
use crate::history_store::{HistoryStore, messages_from_python};
use crate::hooks::AgentHooks;
use crate::human_input::{
    CallLog, HumanInput, SharedCallLog, approval_required, check_reply, new_request, pending_requests,
    reply_from_python, reply_to, unanswered_note, waiting_calls,
};
use crate::json_convert::python_to_json;
use crate::messages::{Message, Reply, RequestKind, ToolCall, UserContent, prompt_from_python};
use crate::model_api::{Api, ModelRequest, last_response_id, stream_model_turn};
use crate::run::Run;
use crate::run_context::Context;
use crate::text_chunks::{TextChunking, TextOutput};
use crate::tools::{ToolDefinition, ToolImplementation, ToolResult, error_result, success_result};
use crate::ws_pool::RunSocket;

/// How many events can wait in the queue before the agent loop pauses.
/// A limit keeps memory bounded if Python reads events slowly.
const EVENT_QUEUE_CAPACITY: usize = 64;

/// The configuration shared (read-only) by every run of an agent.
struct AgentConfig {
    client: ClientSettings,
    /// Chat Completions or Responses, with its settings.
    api: Api,
    model: String,
    system_prompt: String,
    /// `Arc` (a shared pointer) so each run can list them with its own frontend tools.
    tools: Vec<Arc<ToolDefinition>>,
    model_parameters: Map<String, Value>,
    max_turns: u32,
    /// How streamed text is grouped into `TextDelta` events (can be changed per run).
    text_chunks: TextChunking,
    history_store: Option<HistoryStore>,
    hooks: AgentHooks,
    unanswered: Unanswered,
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
    config: Arc<AgentConfig>,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (
        *, client, model, prompt = String::new(), tools = Vec::new(), model_parameters = None, max_turns = 20,
        text_chunks = None, history_store = None, hooks = Vec::new(),
        api = "chat", store = false, encrypted_reasoning = true, unanswered = "close",
    ))]
    #[allow(clippy::too_many_arguments)] // mirrors the keyword arguments of `Agent(...)`
    fn new(
        client: PyRef<'_, Client>,
        model: String,
        prompt: String,
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

        Ok(Engine {
            config: Arc::new(AgentConfig {
                client: client.settings.clone(),
                api,
                model,
                system_prompt: prompt,
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
    /// `frontend_tools` adds tools the app runs, for this run only (dicts with
    /// `name`, `description` and `parameters`, as AG-UI clients send them).
    #[pyo3(signature = (
        prompt = None, *, history = None, conversation_id = None, text_chunks = None, deps = None,
        previous_response_id = None, answers = None, on_request = None, frontend_tools = None,
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
        frontend_tools: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Run> {
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
        let frontend_tools = frontend_tools
            .iter()
            .flatten()
            .map(ToolDefinition::frontend_from_python)
            .collect::<PyResult<Vec<_>>>()?;
        for tool in &frontend_tools {
            if self.config.tools.iter().any(|agent_tool| agent_tool.name == tool.name) {
                return Err(PyValueError::new_err(format!(
                    "The frontend tool {:?} has the name of one of the agent's tools",
                    tool.name
                )));
            }
        }
        let prompt = prompt.as_ref().map(prompt_from_python).transpose()?;
        let history = match history {
            Some(history) => messages_from_python(&history)?,
            None => Vec::new(),
        };
        // A run with a conversation_id is checked after loading, in the loop.
        if prompt.is_none() && history.is_empty() && conversation_id.is_none() {
            return Err(PyValueError::new_err("Give a prompt, a history, or both"));
        }

        let (event_sender, event_receiver) = channel(EVENT_QUEUE_CAPACITY);
        let inputs = RunInputs {
            config: self.config.clone(),
            history,
            prompt,
            conversation_id,
            previous_response_id,
            text_chunks,
            deps: deps.unwrap_or_else(|| py.None()),
            answers: replies,
            on_request,
            frontend_tools,
            event_sender,
        };
        Ok(Run::new(inputs, event_receiver))
    }
}

/// What a run needs to start. Kept aside until the first event is requested.
pub struct RunInputs {
    config: Arc<AgentConfig>,
    text_chunks: TextChunking,
    /// History given directly with `execute(history=...)`.
    history: Vec<Message>,
    prompt: Option<UserContent>,
    /// Set when the history is loaded from (and saved to) the history store.
    conversation_id: Option<String>,
    /// The stored response this run continues, from `execute(previous_response_id=...)`.
    previous_response_id: Option<String>,
    /// Given to tools and hooks as `ctx.deps`.
    deps: Py<PyAny>,
    /// Replies to the tool calls the last run left waiting, by call id.
    answers: HashMap<String, Reply>,
    /// Python function replying to requests while the run waits.
    on_request: Option<Py<PyAny>>,
    /// Tools the app runs, sent with this run only (`execute(frontend_tools=...)`).
    frontend_tools: Vec<ToolDefinition>,
    event_sender: Sender<PyResult<AgentEvent>>,
}

/// Runs the agent loop and makes sure any failure reaches Python as an exception.
pub async fn run_in_background(inputs: RunInputs, python_task_locals: TaskLocals) {
    let event_sender = inputs.event_sender.clone();
    let mut usage = Usage::default();
    let loop_result = AssertUnwindSafe(run_agent_loop(inputs, python_task_locals, &mut usage))
        .catch_unwind()
        .await;

    let error = match loop_result {
        Ok(Ok(())) => return,
        Ok(Err(python_error)) => with_usage(python_error, usage),
        // A panic is a bug in this library; report it instead of ending silently.
        Err(_) => PanicException::new_err("pyox-agents crashed while running the agent (this is a bug)"),
    };
    let _ = event_sender.send(Err(error)).await;
}

/// `usage` is kept up to date as the run goes, so the caller knows the tokens
/// used even when the run fails.
async fn run_agent_loop(inputs: RunInputs, python_task_locals: TaskLocals, usage: &mut Usage) -> PyResult<()> {
    let RunInputs {
        config,
        history,
        prompt,
        conversation_id,
        previous_response_id: run_start_response_id,
        text_chunks,
        deps,
        answers,
        on_request,
        frontend_tools,
        event_sender,
    } = inputs;
    // The tools of this run: the agent's, plus the frontend tools sent with it.
    let tools: Vec<Arc<ToolDefinition>> = config
        .tools
        .iter()
        .cloned()
        .chain(frontend_tools.into_iter().map(Arc::new))
        .collect();
    // The store and the conversation, when this run loads and saves one.
    let stored_conversation = match &conversation_id {
        Some(conversation_id) => config.history_store.as_ref().map(|store| (store, conversation_id)),
        None => None,
    };

    let mut messages = match stored_conversation {
        Some((store, conversation_id)) => store.load(conversation_id, &python_task_locals).await?,
        None => history,
    };
    // Everything from here on is new in this run (and is what gets saved).
    let new_messages_start = messages.len();

    // Tool calls the last run left waiting need answers, or are closed by a new prompt.
    let waiting_count = waiting_calls(&messages).len();
    if waiting_count > 0 && answers.is_empty() {
        if prompt.is_none() {
            return Err(PyValueError::new_err(format!(
                "{waiting_count} tool call(s) are waiting for a person: pass `answers={{call_id: ...}}`"
            )));
        }
        let closed_calls = close_waiting_calls(&messages, config.unanswered)?;
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
    let mut text_output = TextOutput::new(text_chunks, event_sender.clone());
    let hooks = &config.hooks;
    // What tools and hooks see as `ctx`; updated as the run goes.
    let mut context = Context {
        run: pyox_tools::hooks::RunContext {
            conversation_id: conversation_id.clone(),
            ..Default::default()
        },
        deps: Arc::new(deps),
        human_input: Arc::new(HumanInput {
            on_request,
            event_sender: event_sender.clone(),
        }),
    };
    hooks.before_run(&context, &python_task_locals).await?;
    // The WebSocket connection of this run (`Client(transport="websocket")`).
    // It is closed when this function returns, unless given back to the pool.
    let mut run_socket = RunSocket::default();

    // Finishes the run: `after_run` hooks, saving the new messages, then the
    // `RunFinished` event. (A closure: like a small local function that can
    // use the variables around it.)
    let finish = async |finished: RunFinished, context: &Context| -> PyResult<()> {
        hooks.after_run(context, &finished, &python_task_locals).await?;
        if let Some((store, conversation_id)) = stored_conversation {
            store
                .save(conversation_id, finished.added_messages(), &python_task_locals)
                .await?;
        }
        let _ = event_sender.send(Ok(AgentEvent::RunFinished(finished))).await;
        Ok(())
    };

    if !answers.is_empty() {
        let resumed = resume_waiting_calls(&config, &tools, &messages, answers, &context, &python_task_locals).await?;
        add_tool_messages(resumed, &mut messages, &event_sender).await;
        let pending = pending_requests(&messages);
        if !pending.is_empty() {
            let finished = RunFinished {
                text: String::new(),
                turns: 0,
                usage: *usage,
                response_id: None,
                messages,
                new_messages_start,
                pending,
            };
            return finish(finished, &context).await;
        }
    }

    for turn in 1..=config.max_turns {
        context.run.turn = turn;
        // Hooks may change the messages of this one request; the history stays as it is.
        let hooked_messages = hooks
            .before_model_request(&context, &messages, &python_task_locals)
            .await?;
        // The request can continue from the newest response and send only the
        // messages after it, when OpenAI still has that response (the adapter
        // decides). If hooks changed the messages, OpenAI's copy no longer
        // matches them: send them all (on top of the response the run started
        // from, if any).
        let previous_response_id = if hooked_messages.is_some() {
            run_start_response_id.as_deref()
        } else {
            last_response_id(&messages).or(run_start_response_id.as_deref())
        };
        let request = ModelRequest {
            model: &config.model,
            system_prompt: &config.system_prompt,
            messages: hooked_messages.as_ref().unwrap_or(&messages),
            tools: &tools,
            model_parameters: &config.model_parameters,
            previous_response_id,
        };
        let assistant_turn =
            stream_model_turn(&config.client, config.api, &request, &mut run_socket, &mut text_output).await?;
        text_output.finish().await; // the message ended: send any text still grouped
        usage.add(assistant_turn.usage);
        context.run.usage = *usage;

        let assistant_message = Message::Assistant {
            content: assistant_turn.text,
            tool_calls: assistant_turn.tool_calls,
            reasoning: assistant_turn.reasoning,
            response_id: assistant_turn.response_id,
            model: Some(config.model.clone()),
        };
        let assistant_message = hooks
            .after_model_response(&context, assistant_message, &python_task_locals)
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
                };
                let _ = event_sender.send(Ok(AgentEvent::ToolStarted(started))).await;
            }

            // Run all requested tools at the same time and wait for all of them.
            let tool_messages = join_all(
                tool_calls
                    .iter()
                    .map(|call| run_tool_call(&config, &tools, call, None, &context, &python_task_locals)),
            )
            .await
            .into_iter()
            .collect::<PyResult<Vec<_>>>()?;
            add_tool_messages(tool_messages, &mut messages, &event_sender).await;
        }

        let pending = pending_requests(&messages);
        if tool_calls.is_empty() || !pending.is_empty() {
            // The last answer arrived in full, so the connection is clean: the next run may use it.
            run_socket.give_back(&config.client);
            let finished = RunFinished {
                text,
                turns: turn,
                usage: *usage,
                response_id,
                messages,
                new_messages_start,
                pending,
            };
            return finish(finished, &context).await;
        }
    }

    Err(MaxTurnsExceededError::new_err(format!(
        "Stopped after {} turns",
        config.max_turns
    )))
}

/// Adds the tool messages to the history, with a `ToolFinished` event for
/// each call that finished (a waiting one has no result yet).
async fn add_tool_messages(
    tool_messages: Vec<Message>,
    messages: &mut Vec<Message>,
    event_sender: &Sender<PyResult<AgentEvent>>,
) {
    for message in tool_messages {
        if let Message::Tool {
            tool_call_id,
            name,
            content,
            is_error,
            metadata,
            requests,
            ..
        } = &message
            && !message.is_waiting()
        {
            let finished = ToolFinished {
                call_id: tool_call_id.clone(),
                name: name.clone(),
                result: content.clone(),
                is_error: *is_error,
                metadata: metadata.clone(),
                from_app: requests.iter().any(|request| request.kind == RequestKind::Frontend),
            };
            let _ = event_sender.send(Ok(AgentEvent::ToolFinished(finished))).await;
        }
        messages.push(message);
    }
}

/// Runs again the waiting calls that got a reply in `answers`, at the same
/// time. Returns their new tool messages (finished, or waiting again).
async fn resume_waiting_calls(
    config: &AgentConfig,
    tools: &[Arc<ToolDefinition>],
    messages: &[Message],
    mut answers: HashMap<String, Reply>,
    context: &Context,
    python_task_locals: &TaskLocals,
) -> PyResult<Vec<Message>> {
    let waiting = waiting_calls(messages);
    if let Some(call_id) = answers
        .keys()
        .find(|call_id| !waiting.iter().any(|(call, _)| call.id == **call_id))
    {
        return Err(PyValueError::new_err(format!(
            "Tool call {call_id:?} is not waiting for an answer"
        )));
    }

    let mut answered_calls = Vec::new();
    for (call, waiting_message) in waiting {
        let Some(reply) = answers.remove(&call.id) else {
            continue; // not answered yet: it stays waiting
        };
        // Put the reply on the request the call is waiting on.
        let mut message = waiting_message.clone();
        let Message::Tool { requests, .. } = &mut message else {
            unreachable!("waiting calls always have a tool message");
        };
        let request = requests
            .iter_mut()
            .find(|request| request.answer.is_none())
            .expect("a waiting message has an unanswered request");
        check_reply(request.kind, &reply)?;
        request.answer = Some(reply);
        answered_calls.push((call, message));
    }

    join_all(
        answered_calls
            .iter()
            .map(|(call, message)| run_tool_call(config, tools, call, Some(message), context, python_task_locals)),
    )
    .await
    .into_iter()
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

/// Runs one tool call and returns its tool message, which is *waiting* when
/// a request got no reply. `earlier` is the call's waiting message when a
/// resumed run continues it: its replies and saved steps are reused.
async fn run_tool_call(
    config: &AgentConfig,
    tools: &[Arc<ToolDefinition>],
    call: &ToolCall,
    earlier: Option<&Message>,
    context: &Context,
    python_task_locals: &TaskLocals,
) -> PyResult<Message> {
    let call_log = CallLog::new(call.clone(), earlier);
    let result = tool_call_result(config, tools, &call_log, context, python_task_locals).await?;
    // `None` means waiting: no result yet.
    let result = result.unwrap_or_else(|| success_result(String::new(), Value::Null));

    let call_log = call_log.lock().unwrap();
    Ok(Message::Tool {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        content: result.content,
        is_error: result.is_error,
        metadata: result.metadata,
        requests: call_log.requests.clone(),
        steps: call_log.steps.clone(),
    })
}

/// The result of one tool call, or `None` when it waits for a person.
///
/// In order: `before_tool_call` hooks, approval (if the tool or a hook asks
/// for it), the tool itself (or the app, for a frontend tool), then
/// `after_tool_call` hooks. A hook that fails blocks the call: the model gets
/// the error as the result.
async fn tool_call_result(
    config: &AgentConfig,
    tools: &[Arc<ToolDefinition>],
    call_log: &SharedCallLog,
    context: &Context,
    python_task_locals: &TaskLocals,
) -> PyResult<Option<ToolResult>> {
    let hooks = &config.hooks;
    let model_call = call_log.lock().unwrap().call.clone();
    let mut approval_request = None;
    let mut call = match hooks
        .before_tool_call(context, model_call.clone(), python_task_locals)
        .await
    {
        Ok(call) => call,
        // A hook raising `ApprovalRequired` does not block the call: it asks first.
        Err(error) => match approval_required(&error)? {
            Some(request) => {
                approval_request = Some(request);
                model_call
            }
            None => return Ok(Some(error_result(message_for_model(error)))),
        },
    };
    let Some(tool) = tools.iter().find(|tool| tool.name == call.name) else {
        return Ok(Some(error_result(format!("Unknown tool {:?}", call.name))));
    };
    call_log.lock().unwrap().call = call.clone();

    if approval_request.is_none() {
        approval_request = match tool.approval_request(call_log, context, python_task_locals).await {
            Ok(request) => request,
            Err(error) => return Ok(Some(error)),
        };
    }
    if let Some(request) = approval_request {
        match reply_to(call_log, request, &context.human_input, python_task_locals).await? {
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

    let result = match tool.implementation {
        // The app runs a frontend tool: its reply is the result.
        ToolImplementation::Frontend => {
            let request = new_request(RequestKind::Frontend, call.arguments.clone(), Value::Null);
            match reply_to(call_log, request, &context.human_input, python_task_locals).await? {
                None => return Ok(None),
                Some(Reply::Answer { value, metadata }) => success_result(text_for_model(value), metadata),
                Some(Reply::Deny { message, .. }) => error_result(message),
                Some(Reply::Approve { .. }) => unreachable!("check_reply only lets Answer or Deny through"),
            }
        }
        _ => {
            let result = tool.run(&call, call_log, context, python_task_locals).await;
            // A `ctx.ask` inside the tool may have stopped it.
            let call_log = call_log.lock().unwrap();
            if call_log.is_waiting() {
                return Ok(None);
            }
            match &call_log.denied {
                Some(message) => error_result(message.clone()),
                None => result,
            }
        }
    };

    match hooks.after_tool_call(context, &call, result, python_task_locals).await {
        Ok(result) => Ok(Some(result)),
        Err(error) => Ok(Some(error_result(message_for_model(error)))),
    }
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
