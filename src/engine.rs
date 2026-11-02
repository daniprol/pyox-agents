//! `Engine`: an agent's configuration, checked once when the `Agent` is
//! created. It never changes after that, so one engine serves many runs at
//! the same time. `stream(...)` checks a run's arguments and gives Python a
//! `Run`; the run itself is `agent_loop::run_agent_loop`.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value};
use tokio::sync::mpsc::channel;
use tokio::sync::watch;

use crate::agent_loop::RunInputs;
use crate::client::{Client, ClientSettings};
use crate::errors::InvalidResponseError;
use crate::events::EventSender;
use crate::history_store::{HistoryStore, messages_from_python};
use crate::hooks::AgentHooks;
use crate::human_input::{HumanInput, reply_from_python};
use crate::json_convert::python_to_json;
use crate::messages::content_from_python;
use crate::model_api::Api;
use crate::run::new_run;
use crate::state::RunState;
use crate::telemetry::Telemetry;
use crate::text_chunks::TextChunking;
use crate::tools::ToolDefinition;

/// How many events can wait in the queue before the agent loop pauses.
/// A limit keeps memory bounded if Python reads events slowly.
const EVENT_QUEUE_CAPACITY: usize = 64;

/// The configuration shared (read-only) by every run of an agent.
pub struct AgentConfig {
    /// `Agent(name=...)`: names the run's span (see `telemetry`).
    pub name: Option<String>,
    pub client: ClientSettings,
    /// Chat Completions or Responses, with its settings.
    pub api: Api,
    pub model: String,
    /// The system prompt (`Agent(instructions=...)`).
    pub instructions: String,
    /// `Arc` (a shared pointer) so each run can list them with its own frontend tools.
    pub tools: Vec<Arc<ToolDefinition>>,
    pub model_parameters: Map<String, Value>,
    pub max_turns: u32,
    /// How streamed text is grouped into `TextDelta` events (can be changed per run).
    pub text_chunks: TextChunking,
    pub history_store: Option<HistoryStore>,
    pub hooks: AgentHooks,
    pub unanswered: Unanswered,
    pub output: Option<StructuredOutput>,
}

/// `Agent(output_type=...)`: the answer is JSON matching a schema, turned into a Python value.
pub struct StructuredOutput {
    /// `{"name": ..., "schema": ..., "strict": true}`, sent with every request.
    pub schema: Value,
    /// A Python function: the answer's text in, the value out (made by `Agent`).
    parse: Py<PyAny>,
}

impl StructuredOutput {
    /// The answer as a Python value. The model wrote something else than the
    /// schema asks for (rare in strict mode): `InvalidResponseError`.
    pub fn parse(&self, text: &str) -> PyResult<Py<PyAny>> {
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
pub enum Unanswered {
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
    #[allow(clippy::too_many_arguments)] // mirrors the keyword arguments of `Agent.stream(...)`
    fn start(
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
