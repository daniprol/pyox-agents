//! The agent loop.
//!
//! - `Engine` holds the agent's configuration. It never changes after creation,
//!   so one engine can serve many runs at the same time.
//! - `Run` is one execution. It owns its history and its stream of events.
//!
//! The loop (in `run_agent_loop`):
//!
//! ```text
//! load the conversation from the history store (if the run names one)
//! before_run hooks
//! repeat until the model stops calling tools (at most `max_turns` times):
//!     before_model_request hooks
//!     stream a model response, forwarding text to Python as it arrives
//!     after_model_response hooks, then add the assistant message to the history
//!     run the requested tools (concurrently) and add their results;
//!         each call goes through before_tool_call and after_tool_call hooks
//! after_run hooks
//! save the run's new messages to the history store (if any)
//! ```
//!
//! Every step only touches the hooks when the agent has some (see `hooks`).

use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use futures::future::join_all;
use pyo3::exceptions::{PyStopAsyncIteration, PyValueError};
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_async_runtimes::TaskLocals;
use serde_json::{Map, Value};
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tokio::task::JoinHandle;

use crate::client::{Client, ClientSettings};
use crate::errors::{HookError, MaxTurnsExceededError};
use crate::events::{AgentEvent, RunFinished, ToolFinished, ToolStarted};
use crate::history_store::{HistoryStore, messages_from_python};
use crate::hooks::AgentHooks;
use crate::json_convert::python_to_json;
use crate::messages::{Message, TokenUsage, ToolCall, UserContent, prompt_from_python};
use crate::openai_chat::{build_request_body, stream_chat_completion};
use crate::run_context::Context;
use crate::text_chunks::{TextChunking, TextOutput};
use crate::tools::{ToolDefinition, ToolResult, error_result};

/// How many events can wait in the queue before the agent loop pauses.
/// A limit keeps memory bounded if Python reads events slowly.
const EVENT_QUEUE_CAPACITY: usize = 64;

/// The configuration shared (read-only) by every run of an agent.
struct AgentConfig {
    client: ClientSettings,
    model: String,
    system_prompt: String,
    tools: Vec<ToolDefinition>,
    model_parameters: Map<String, Value>,
    max_turns: u32,
    /// How streamed text is grouped into `TextDelta` events (can be changed per run).
    text_chunks: TextChunking,
    history_store: Option<HistoryStore>,
    hooks: AgentHooks,
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
    ) -> PyResult<Self> {
        let tools = tools
            .iter()
            .map(ToolDefinition::from_python)
            .collect::<PyResult<Vec<_>>>()?;
        for (position, tool) in tools.iter().enumerate() {
            if tools[..position].iter().any(|earlier| earlier.name == tool.name) {
                return Err(PyValueError::new_err(format!("Two tools are named {:?}", tool.name)));
            }
        }

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
            }),
        })
    }

    /// Starts a run. Nothing is sent until Python starts iterating over it.
    ///
    /// The history comes from `history` (message dicts or JSON text), or is
    /// loaded from the agent's history store when `conversation_id` is given.
    /// `deps` is any object, given to tools and hooks as `ctx.deps`.
    #[pyo3(signature = (prompt = None, *, history = None, conversation_id = None, text_chunks = None, deps = None))]
    fn execute(
        &self,
        py: Python<'_>,
        prompt: Option<Bound<'_, PyAny>>,
        history: Option<Bound<'_, PyAny>>,
        conversation_id: Option<String>,
        text_chunks: Option<Bound<'_, PyAny>>,
        deps: Option<Py<PyAny>>,
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
        Ok(Run {
            not_started: Mutex::new(Some(RunInputs {
                config: self.config.clone(),
                history,
                prompt,
                conversation_id,
                text_chunks,
                deps: deps.unwrap_or_else(|| py.None()),
                event_sender,
            })),
            event_receiver: Arc::new(tokio::sync::Mutex::new(event_receiver)),
            background_task: Mutex::new(None),
        })
    }
}

/// What a run needs to start. Kept aside until the first event is requested.
struct RunInputs {
    config: Arc<AgentConfig>,
    text_chunks: TextChunking,
    /// History given directly with `execute(history=...)`.
    history: Vec<Message>,
    prompt: Option<UserContent>,
    /// Set when the history is loaded from (and saved to) the history store.
    conversation_id: Option<String>,
    /// Given to tools and hooks as `ctx.deps`.
    deps: Py<PyAny>,
    event_sender: Sender<PyResult<AgentEvent>>,
}

/// Python: `async for event in run: ...`
///
/// The agent loop runs as a background Rust task and sends events through a
/// queue; each `__anext__` call waits for the next one.
#[pyclass(frozen, module = "pyox_agents")]
pub struct Run {
    not_started: Mutex<Option<RunInputs>>,
    event_receiver: Arc<tokio::sync::Mutex<Receiver<PyResult<AgentEvent>>>>,
    background_task: Mutex<Option<JoinHandle<()>>>,
}

#[pymethods]
impl Run {
    fn __aiter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(inputs) = self.not_started.lock().unwrap().take() {
            // Remember Python's event loop: Python tools must run on it.
            let python_task_locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
            let task = pyo3_async_runtimes::tokio::get_runtime().spawn(run_in_background(inputs, python_task_locals));
            *self.background_task.lock().unwrap() = Some(task);
        }

        let event_receiver = self.event_receiver.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match event_receiver.lock().await.recv().await {
                Some(Ok(event)) => Python::attach(|py| event.into_python(py)),
                Some(Err(error)) => Err(error),
                None => Err(PyStopAsyncIteration::new_err(())), // the run is over
            }
        })
    }
}

impl Drop for Run {
    /// If Python stops iterating early (e.g. `break`), stop the agent loop too.
    fn drop(&mut self) {
        if let Some(task) = self.background_task.lock().unwrap().take() {
            task.abort();
        }
    }
}

/// Runs the agent loop and makes sure any failure reaches Python as an exception.
async fn run_in_background(inputs: RunInputs, python_task_locals: TaskLocals) {
    let event_sender = inputs.event_sender.clone();
    let loop_result = AssertUnwindSafe(run_agent_loop(inputs, python_task_locals))
        .catch_unwind()
        .await;

    let error = match loop_result {
        Ok(Ok(())) => return,
        Ok(Err(python_error)) => python_error,
        // A panic is a bug in this library; report it instead of ending silently.
        Err(_) => PanicException::new_err("pyox-agents crashed while running the agent (this is a bug)"),
    };
    let _ = event_sender.send(Err(error)).await;
}

async fn run_agent_loop(inputs: RunInputs, python_task_locals: TaskLocals) -> PyResult<()> {
    let RunInputs {
        config,
        history,
        prompt,
        conversation_id,
        text_chunks,
        deps,
        event_sender,
    } = inputs;
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
    if let Some(prompt) = prompt {
        messages.push(Message::User { content: prompt });
    }
    if messages.is_empty() {
        return Err(PyValueError::new_err(format!(
            "Conversation {:?} has no messages yet: give a prompt",
            conversation_id.unwrap_or_default()
        )));
    }
    let mut total_usage = TokenUsage::default();
    let mut text_output = TextOutput::new(text_chunks, event_sender.clone());
    let hooks = &config.hooks;
    // What tools and hooks see as `ctx`; updated as the run goes.
    let mut context = Context {
        run: pyox_tools::hooks::RunContext {
            conversation_id: conversation_id.clone(),
            ..Default::default()
        },
        deps: Arc::new(deps),
    };
    hooks.before_run(&context, &python_task_locals).await?;

    for turn in 1..=config.max_turns {
        context.run.turn = turn;
        // Hooks may change the messages of this one request; the history stays as it is.
        let hooked_messages = hooks
            .before_model_request(&context, &messages, &python_task_locals)
            .await?;
        let request_body = build_request_body(
            &config.model,
            &config.system_prompt,
            hooked_messages.as_ref().unwrap_or(&messages),
            &config.tools,
            &config.model_parameters,
        );
        let assistant_turn = stream_chat_completion(&config.client, &request_body, &mut text_output).await?;
        text_output.finish().await; // the message ended: send any text still grouped
        total_usage.add(assistant_turn.usage);
        context.run.input_tokens = total_usage.input_tokens;
        context.run.output_tokens = total_usage.output_tokens;

        let assistant_message = Message::Assistant {
            content: assistant_turn.text,
            tool_calls: assistant_turn.tool_calls,
        };
        let assistant_message = hooks
            .after_model_response(&context, assistant_message, &python_task_locals)
            .await?;
        let Message::Assistant { content, tool_calls } = &assistant_message else {
            unreachable!("after_model_response always gives an assistant message");
        };
        let (text, tool_calls) = (content.clone(), tool_calls.clone());
        messages.push(assistant_message);

        if tool_calls.is_empty() {
            let finished = RunFinished {
                text,
                turns: turn,
                input_tokens: total_usage.input_tokens,
                output_tokens: total_usage.output_tokens,
                messages,
                new_messages_start,
            };
            hooks.after_run(&context, &finished, &python_task_locals).await?;
            if let Some((store, conversation_id)) = stored_conversation {
                store
                    .save(conversation_id, finished.added_messages(), &python_task_locals)
                    .await?;
            }
            let _ = event_sender.send(Ok(AgentEvent::RunFinished(finished))).await;
            return Ok(());
        }

        for call in &tool_calls {
            let started = ToolStarted {
                call_id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            };
            let _ = event_sender.send(Ok(AgentEvent::ToolStarted(started))).await;
        }

        // Run all requested tools at the same time and wait for all of them.
        let outputs = join_all(
            tool_calls
                .iter()
                .map(|call| run_tool_call(&config, call, &context, &python_task_locals)),
        )
        .await;

        for (call, output) in tool_calls.into_iter().zip(outputs) {
            let finished = ToolFinished {
                call_id: call.id.clone(),
                name: call.name.clone(),
                result: output.content.clone(),
                is_error: output.is_error,
            };
            let _ = event_sender.send(Ok(AgentEvent::ToolFinished(finished))).await;
            messages.push(Message::Tool {
                tool_call_id: call.id,
                name: call.name,
                content: output.content,
                is_error: output.is_error,
            });
        }
    }

    Err(MaxTurnsExceededError::new_err(format!(
        "Stopped after {} turns",
        config.max_turns
    )))
}

/// Runs one tool call, through the `before_tool_call` and `after_tool_call` hooks.
/// A hook that fails blocks the call: the model gets the error as the result.
async fn run_tool_call(
    config: &AgentConfig,
    call: &ToolCall,
    context: &Context,
    python_task_locals: &TaskLocals,
) -> ToolResult {
    let hooks = &config.hooks;
    let call = match hooks.before_tool_call(context, call.clone(), python_task_locals).await {
        Ok(call) => call,
        Err(error) => return error_result(message_for_model(error)),
    };

    let result = match config.tools.iter().find(|tool| tool.name == call.name) {
        Some(tool) => tool.run(&call, context, python_task_locals).await,
        None => error_result(format!("Unknown tool {:?}", call.name)),
    };

    match hooks.after_tool_call(context, &call, result, python_task_locals).await {
        Ok(result) => result,
        Err(error) => error_result(message_for_model(error)),
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
