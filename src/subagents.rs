//! Subagents: an agent the model calls like a tool.
//!
//! Python: `writer = Agent(tools=[researcher.as_tool("research", "Research a topic")])`.
//!
//! The model calls the tool with a `task`. The subagent runs its own agent
//! loop with a fresh conversation (the task is its first message), and its
//! final answer is the tool's result. While it works:
//!
//! - its events come in the run's stream, marked with its id (the id of the
//!   tool call), between `SubagentStarted` and `SubagentFinished`;
//! - its tools share the run's `deps`, and the run's state if allowed;
//! - its approvals and `ctx.ask` questions go to the person, exactly like the
//!   main agent's, with ids like `call_0/call_1` (see `human_input`). While
//!   they wait, the main agent's call waits too, and keeps the subagent's
//!   conversation in its tool message (`subagent`), to continue it once
//!   answered;
//! - its tokens are added to the run's usage.
//!
//! What the tool allows (`as_tool(...)` options):
//!
//! | Option          | Values                     | Meaning                                         |
//! |-----------------|----------------------------|-------------------------------------------------|
//! | `inherit_tools` | `False`, `True`            | also give it the calling agent's tools          |
//! | `talk_to`       | `None`, `"user"`, `"parent"` | who its model can ask questions (see `TalkTo`) |
//! | `state`         | `None`, `"read"`, `"write"` | what it may do with the run's state            |
//!
//! A subagent cannot have subagents of its own: one level keeps runs easy to follow.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyox_tools::usage::Usage;
use serde_json::{Value, json};

use crate::agent::{AgentConfig, Engine, RunInputs, ToolRun, run_agent_loop};
use crate::errors::AgentError;
use crate::events::{AgentEvent, SubagentFinished, SubagentStarted};
use crate::human_input::{HumanInput, SharedCallLog};
use crate::messages::{Message, Reply, ToolCall};
use crate::state::StateAccess;
use crate::tools::{Approval, ToolDefinition, ToolImplementation, ToolResult, error_result, success_result};

/// The name of the tool a subagent with `talk_to="user"` asks the person with.
const ASK_USER: &str = "ask_user";
/// The name of the tool a subagent with `talk_to="parent"` asks the calling agent with.
const ASK_PARENT: &str = "ask_parent";

/// A tool that runs another agent.
pub struct Subagent {
    /// The subagent's own configuration: model, prompt, tools, hooks...
    agent: Arc<AgentConfig>,
    inherit_tools: bool,
    talk_to: TalkTo,
    state: StateAccess,
}

/// Who a subagent's model can ask questions. (Approvals and `ctx.ask` in its
/// tools always go to the person.)
#[derive(Clone, Copy, PartialEq)]
enum TalkTo {
    /// Nobody: it does the task and answers.
    Nobody,
    /// The person, with an `ask_user(question)` tool.
    User,
    /// The calling agent, with an `ask_parent(question)` tool. The subagent
    /// stops, and the calling agent reads the question as the tool's result.
    /// It answers (or follows up later) by calling the tool again with the
    /// `subagent_id` from that result: the subagent goes on with its conversation.
    Parent,
}

/// Reads a `pyox_agents.Subagent` (made by `Agent.as_tool(...)`). `None` if
/// `tool` is something else.
pub fn subagent_tool_from_python(tool: &Bound<'_, PyAny>) -> PyResult<Option<ToolDefinition>> {
    let Ok(agent) = tool.getattr("agent") else {
        return Ok(None);
    };
    let Ok(engine) = agent.getattr("_engine") else {
        return Ok(None);
    };
    let Ok(engine) = engine.cast::<Engine>() else {
        return Ok(None);
    };
    let agent = engine.get().config.clone();
    if agent
        .tools
        .iter()
        .any(|tool| matches!(tool.implementation, ToolImplementation::Subagent(_)))
    {
        return Err(PyValueError::new_err(
            "A subagent cannot have subagents of its own: give their tools to the main agent",
        ));
    }
    let talk_to = match tool.getattr("talk_to")?.extract::<Option<String>>()?.as_deref() {
        None => TalkTo::Nobody,
        Some("user") => TalkTo::User,
        Some("parent") => TalkTo::Parent,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "Unknown talk_to={other:?}: use None, \"user\" or \"parent\""
            )));
        }
    };
    let subagent = Subagent {
        agent,
        inherit_tools: tool.getattr("inherit_tools")?.is_truthy()?,
        talk_to,
        state: StateAccess::from_python(&tool.getattr("state")?)?,
    };
    Ok(Some(ToolDefinition {
        name: tool.getattr("name")?.extract()?,
        description: tool.getattr("description")?.extract()?,
        parameters: subagent.parameters(),
        implementation: ToolImplementation::Subagent(subagent),
        approval: Approval::Never,
    }))
}

impl Subagent {
    /// The JSON schema of the tool's arguments.
    fn parameters(&self) -> Value {
        let mut parameters = json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What to do, with everything needed to do it: the subagent sees nothing else",
                },
            },
            "required": ["task"],
        });
        if self.talk_to == TalkTo::Parent {
            parameters["properties"]["subagent_id"] = json!({
                "type": "string",
                "description": "To answer the subagent or follow up: the subagent_id from its earlier result",
            });
        }
        parameters
    }

    /// The tools the subagent gets for this run, on top of its own.
    fn run_tools(&self, calling_agent_tools: &[Arc<ToolDefinition>]) -> Vec<Arc<ToolDefinition>> {
        let mut tools = Vec::new();
        if self.inherit_tools {
            for tool in calling_agent_tools {
                let is_subagent = matches!(tool.implementation, ToolImplementation::Subagent(_));
                let has_its_own = self.agent.tools.iter().any(|own| own.name == tool.name);
                if !is_subagent && !has_its_own {
                    tools.push(tool.clone());
                }
            }
        }
        match self.talk_to {
            TalkTo::Nobody => {}
            TalkTo::User => tools.push(Arc::new(ask_tool(
                ASK_USER,
                "Ask the user a question and wait for the answer",
                ToolImplementation::AskUser,
            ))),
            TalkTo::Parent => tools.push(Arc::new(ask_tool(
                ASK_PARENT,
                "Ask the agent that gave you this task a question. You stop here: its answer comes as the next message",
                ToolImplementation::AskParent,
            ))),
        }
        tools
    }

    /// Runs the subagent for one tool call. `None` means it waits for a person.
    pub async fn run(&self, call_log: &SharedCallLog, run: &ToolRun<'_>) -> PyResult<Option<ToolResult>> {
        let call = call_log.lock().unwrap().call.clone();
        let Some(Value::String(task)) = call.arguments.get("task") else {
            return Ok(Some(error_result("Give the subagent a `task`")));
        };
        let task = task.clone();

        // Where the subagent's conversation starts.
        let (history, prompt, answers) = {
            let mut log = call_log.lock().unwrap();
            let continued_id = match call.arguments.get("subagent_id") {
                Some(Value::String(subagent_id)) if self.talk_to == TalkTo::Parent => Some(subagent_id),
                _ => None,
            };
            if log.subagent.is_empty() && !log.subagent_answers.is_empty() {
                return Err(PyValueError::new_err(
                    "The history has no conversation for this subagent, so its requests cannot be answered. \
                     Keep the conversation on the server (`conversation_id=`) for subagents that wait for a person.",
                ));
            }
            if !log.subagent.is_empty() {
                // This call waited for a person, who answered: go on with the answers.
                (log.subagent.clone(), None, std::mem::take(&mut log.subagent_answers))
            } else if let Some(subagent_id) = continued_id {
                let Some(conversation) = earlier_conversation(run.history, &call.name, subagent_id) else {
                    return Ok(Some(error_result(format!(
                        "There is no conversation with subagent_id {subagent_id:?}"
                    ))));
                };
                (conversation, Some(task.clone().into()), HashMap::new())
            } else {
                (Vec::new(), Some(task.clone().into()), HashMap::new())
            }
        };

        let events = run.context.events();
        let started = SubagentStarted {
            call_id: call.id.clone(),
            name: call.name.clone(),
            task,
            subagent_id: None, // set by the event sender
        };
        events.send(AgentEvent::SubagentStarted(started)).await;

        let on_request = Python::attach(|py| {
            let on_request = run.context.human_input.on_request.as_ref();
            on_request.map(|function| function.clone_ref(py))
        });
        let inputs = RunInputs {
            config: self.agent.clone(),
            text_chunks: run.context.text_chunks,
            history,
            prompt,
            conversation_id: None,
            previous_response_id: None,
            deps: run.context.deps.clone(),
            answers,
            human_input: Arc::new(HumanInput {
                on_request,
                events: events.for_subagent(&call.id),
            }),
            extra_tools: self.run_tools(run.tools),
            state: run.context.state.with_access(self.state),
        };
        let mut usage = Usage::default();
        // `Box::pin`: the subagent's loop runs inside this one (Rust needs a box for that).
        let outcome = Box::pin(run_agent_loop(inputs, run.python_task_locals.clone(), &mut usage)).await;
        call_log.lock().unwrap().subagent_usage = usage;

        let finished = match outcome {
            Ok(finished) => finished,
            // An API error, too many turns...: the calling agent can react to it.
            Err(error) if Python::attach(|py| error.is_instance_of::<AgentError>(py)) => {
                let message = Python::attach(|py| error.value(py).to_string());
                let result = format!("The subagent failed: {message}");
                self.send_finished(&call, &result, true, false, usage, run).await;
                return Ok(Some(error_result(result)));
            }
            Err(error) => return Err(error),
        };

        let waiting = !finished.pending.is_empty();
        let result = if waiting {
            String::new()
        } else {
            self.result_for_model(&call, &finished.text, &finished.messages)
        };
        call_log.lock().unwrap().subagent = finished.messages;
        self.send_finished(&call, &result, false, waiting, usage, run).await;
        Ok((!waiting).then(|| success_result(result, Value::Null)))
    }

    /// What the calling agent reads: the subagent's answer or, with
    /// `talk_to="parent"`, its question, and how to reply.
    fn result_for_model(&self, call: &ToolCall, answer: &str, conversation: &[Message]) -> String {
        if self.talk_to != TalkTo::Parent {
            return answer.to_string();
        }
        let (tool, id) = (&call.name, &call.id);
        match question_for_parent(conversation) {
            Some(question) => format!(
                "The subagent asks: {question}\n\nTo answer, call {tool} with subagent_id=\"{id}\" and your answer as the task."
            ),
            None => format!("{answer}\n\n(To follow up, call {tool} with subagent_id=\"{id}\".)"),
        }
    }

    async fn send_finished(
        &self,
        call: &ToolCall,
        result: &str,
        is_error: bool,
        waiting: bool,
        usage: Usage,
        run: &ToolRun<'_>,
    ) {
        let finished = SubagentFinished {
            call_id: call.id.clone(),
            name: call.name.clone(),
            result: result.to_string(),
            is_error,
            waiting,
            usage,
            subagent_id: None, // set by the event sender
        };
        run.context.events().send(AgentEvent::SubagentFinished(finished)).await;
    }
}

/// `ask_user` or `ask_parent`: a tool taking one `question`.
fn ask_tool(name: &str, description: &str, implementation: ToolImplementation) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        parameters: json!({
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }),
        implementation,
        approval: Approval::Never,
    }
}

/// The question of an `ask_parent` call, as it reaches the person or the calling agent.
pub fn question_of(call: &ToolCall) -> Value {
    call.arguments.get("question").cloned().unwrap_or(Value::Null)
}

/// True when the model called `ask_parent` this turn: the subagent's run stops there.
pub fn asked_parent(tools: &[Arc<ToolDefinition>], calls: &[ToolCall]) -> bool {
    calls.iter().any(|call| {
        tools
            .iter()
            .any(|tool| tool.name == call.name && matches!(tool.implementation, ToolImplementation::AskParent))
    })
}

/// The question, if the subagent's last turn called `ask_parent`.
fn question_for_parent(conversation: &[Message]) -> Option<String> {
    let Some(Message::Assistant { tool_calls, .. }) = conversation
        .iter()
        .rfind(|message| matches!(message, Message::Assistant { .. }))
    else {
        return None;
    };
    let call = tool_calls.iter().find(|call| call.name == ASK_PARENT)?;
    Some(match question_of(call) {
        Value::String(question) => question,
        other => other.to_string(),
    })
}

/// The conversation of an earlier call to the subagent tool `tool_name`, by its call id.
fn earlier_conversation(history: &[Message], tool_name: &str, subagent_id: &str) -> Option<Vec<Message>> {
    history.iter().rev().find_map(|message| match message {
        Message::Tool {
            tool_call_id,
            name,
            subagent,
            ..
        } if tool_call_id == subagent_id && name == tool_name && !subagent.is_empty() => Some(subagent.clone()),
        _ => None,
    })
}

/// Takes the answers meant for the subagent of the call `call_id`, with the
/// subagent's own ids: `call_0/call_1` becomes `call_1`.
pub fn take_subagent_answers(answers: &mut HashMap<String, Reply>, call_id: &str) -> HashMap<String, Reply> {
    let prefix = format!("{call_id}/");
    let keys: Vec<String> = answers.keys().filter(|key| key.starts_with(&prefix)).cloned().collect();
    keys.into_iter()
        .map(|key| {
            let reply = answers.remove(&key).expect("the key was just found");
            (key[prefix.len()..].to_string(), reply)
        })
        .collect()
}
