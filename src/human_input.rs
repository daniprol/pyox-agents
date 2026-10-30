//! Human in the loop: tool calls that wait for a person.
//!
//! A tool call can make a *request*: an approval before it runs, a question
//! from `ctx.ask(...)` inside it, or (for a frontend tool) its whole result.
//! The reply comes from one of two places:
//!
//! 1. **The history**, when a run resumes. The last run stopped at the
//!    request, the app sent `answers={call_id: reply}`, and the call runs
//!    again from its start. Its earlier requests find their stored replies.
//! 2. **The live handler**, `execute(on_request=...)`: Python answers while
//!    the run waits, and the tool goes on from where it asked.
//!
//! With neither, the call is *waiting*: the run ends and lists it in
//! `RunFinished.pending`. Nothing is stored anywhere but in the history.
//!
//! Python's side (`Answer`, `Approve`, `Deny`, `ApprovalRequired`) lives in
//! `pyox_agents/human_input.py`.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use serde_json::{Map, Value};
use tokio::sync::mpsc::Sender;

use crate::events::AgentEvent;
use crate::json_convert::{json_to_python, python_to_json};
use crate::messages::{Message, Reply, Request, RequestKind, ToolCall};
use crate::python_call::call_python;

/// Where a run sends new requests while it is alive.
pub struct HumanInput {
    /// `execute(on_request=...)`: called with each new request; returns a reply or `None`.
    pub on_request: Option<Py<PyAny>>,
    /// New requests are also sent to Python as `InputRequested` events.
    pub event_sender: Sender<PyResult<AgentEvent>>,
}

/// Everything one tool call asked so far, shared by the agent loop and the
/// tool's `ctx`. `Arc<Mutex<...>>` is how Rust shares something that changes
/// between tasks: `Arc` is a shared reference, `Mutex` lets one user at a time
/// change it (like a `threading.Lock` around the data).
pub type SharedCallLog = Arc<Mutex<CallLog>>;

pub struct CallLog {
    pub call: ToolCall,
    /// Requests from earlier runs of this call (with their replies), then new ones.
    pub requests: Vec<Request>,
    /// Saved `ctx.step(...)` results, by name.
    pub steps: Map<String, Value>,
    /// How many requests this run of the call has made, to replay them in order.
    requests_made: usize,
    /// Step names used in this run, to catch a name used twice.
    steps_used: HashSet<String>,
    /// Set when a question was answered with `Deny`: the call's result.
    pub denied: Option<String>,
}

/// What looking up a request in the log found.
enum Lookup {
    /// An earlier run already got this reply.
    Replied(Reply),
    /// No reply yet; the request is at this position of `requests`.
    New(usize),
}

impl CallLog {
    /// A log for `call`, continuing what an earlier run of the call stored (if any).
    pub fn new(call: ToolCall, earlier: Option<&Message>) -> SharedCallLog {
        let (requests, steps) = match earlier {
            Some(Message::Tool { requests, steps, .. }) => (requests.clone(), steps.clone()),
            _ => (Vec::new(), Map::new()),
        };
        Arc::new(Mutex::new(CallLog {
            call,
            requests,
            steps,
            requests_made: 0,
            steps_used: HashSet::new(),
            denied: None,
        }))
    }

    /// True when a request still has no reply: the call must wait.
    pub fn is_waiting(&self) -> bool {
        self.requests.iter().any(|request| request.answer.is_none())
    }

    /// The reply to the call's approval request, if it has one.
    pub fn approval(&self) -> Option<&Reply> {
        let approval = self
            .requests
            .iter()
            .find(|request| request.kind == RequestKind::Approval)?;
        approval.answer.as_ref()
    }

    /// Finds the stored request matching `request`: the one with the same key,
    /// or else the one at the same position (the n-th request of the call).
    fn look_up(&mut self, request: Request) -> Lookup {
        let position = match &request.key {
            Some(key) => self.requests.iter().position(|stored| stored.key.as_ref() == Some(key)),
            None => Some(self.requests_made).filter(|&position| position < self.requests.len()),
        };
        self.requests_made += 1;
        match position {
            Some(position) => match &self.requests[position].answer {
                Some(reply) => Lookup::Replied(reply.clone()),
                None => {
                    self.requests[position] = request;
                    Lookup::New(position)
                }
            },
            None => {
                self.requests.push(request);
                Lookup::New(self.requests.len() - 1)
            }
        }
    }

    /// The saved result of step `name`, or an error if this run already used the name.
    pub fn start_step(&mut self, name: &str) -> PyResult<Option<Value>> {
        if !self.steps_used.insert(name.to_string()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ctx.step({name:?}) was already used in this tool call: give each step its own name"
            )));
        }
        Ok(self.steps.get(name).cloned())
    }
}

/// Gets the reply to `request`, from the history or the live handler.
/// `None` means there is none yet: the call must wait.
pub async fn reply_to(
    log: &SharedCallLog,
    request: Request,
    human_input: &HumanInput,
    python_task_locals: &TaskLocals,
) -> PyResult<Option<Reply>> {
    // The lock is held only inside this block (it is released at the `}`),
    // never while waiting for Python.
    let kind = request.kind;
    let (position, event) = {
        let mut log = log.lock().unwrap();
        match log.look_up(request) {
            Lookup::Replied(reply) => {
                check_reply(kind, &reply)?;
                return Ok(Some(reply));
            }
            Lookup::New(position) => (position, InputRequested::new(&log.call, &log.requests[position])),
        }
    };
    let _ = human_input
        .event_sender
        .send(Ok(AgentEvent::InputRequested(event.clone())))
        .await;

    let Some(on_request) = &human_input.on_request else {
        return Ok(None);
    };
    let on_request = Python::attach(|py| on_request.clone_ref(py));
    let reply = call_python(
        move |py| on_request.bind(py).call1((Py::new(py, event)?,)),
        python_task_locals,
    )
    .await?;
    let reply = Python::attach(|py| reply_from_python(reply.bind(py)))?;
    if let Some(reply) = &reply {
        check_reply(kind, reply)?;
        log.lock().unwrap().requests[position].answer = Some(reply.clone());
    }
    Ok(reply)
}

/// Reads an `Answer`, `Approve` or `Deny` (or `None`) given by Python.
pub fn reply_from_python(reply: &Bound<'_, PyAny>) -> PyResult<Option<Reply>> {
    if reply.is_none() {
        return Ok(None);
    }
    // Each reply class has `_to_json()`, giving e.g. {"type": "deny", "message": "..."}.
    let Ok(reply_json) = reply.call_method0("_to_json") else {
        return Err(PyTypeError::new_err(format!(
            "Expected Answer, Approve, Deny or None, got {}",
            reply.repr()?
        )));
    };
    match serde_json::from_value(python_to_json(&reply_json)?) {
        Ok(reply) => Ok(Some(reply)),
        Err(error) => Err(PyTypeError::new_err(format!(
            "Invalid reply {}: {error}",
            reply.repr()?
        ))),
    }
}

/// Builds the Python `Answer`, `Approve` or `Deny` for a stored reply.
pub fn reply_to_python<'py>(py: Python<'py>, reply: &Reply) -> PyResult<Bound<'py, PyAny>> {
    let classes = py.import("pyox_agents.human_input")?;
    match reply {
        Reply::Answer { value, metadata } => classes
            .getattr("Answer")?
            .call1((json_to_python(py, value)?, json_to_python(py, metadata)?)),
        Reply::Approve { arguments, metadata } => {
            let arguments = match arguments {
                Some(arguments) => json_to_python(py, arguments)?,
                None => py.None().into_bound(py),
            };
            classes
                .getattr("Approve")?
                .call1((arguments, json_to_python(py, metadata)?))
        }
        Reply::Deny { message, metadata } => classes.getattr("Deny")?.call1((message, json_to_python(py, metadata)?)),
    }
}

/// If `error` is an `ApprovalRequired` raised by a hook, the approval request it asks for.
pub fn approval_required(error: &PyErr) -> PyResult<Option<Request>> {
    Python::attach(|py| {
        let approval_required = py.import("pyox_agents.human_input")?.getattr("ApprovalRequired")?;
        let error = error.value(py);
        if !error.is_instance(&approval_required)? {
            return Ok(None);
        }
        Ok(Some(new_request(
            RequestKind::Approval,
            python_to_json(&error.getattr("data")?)?,
            python_to_json(&error.getattr("metadata")?)?,
        )))
    })
}

/// A request without a key or a reply.
pub fn new_request(kind: RequestKind, data: Value, metadata: Value) -> Request {
    Request {
        kind,
        key: None,
        data,
        metadata,
        answer: None,
    }
}

/// Python: the request a waiting tool call made. Sent as an event when it is
/// made, given to `on_request`, and listed in `RunFinished.pending`.
#[derive(Clone)]
#[pyclass(frozen, module = "pyox_agents")]
pub struct InputRequested {
    /// Answer with `answers={call_id: ...}`.
    #[pyo3(get)]
    pub call_id: String,
    #[pyo3(get)]
    pub tool_name: String,
    pub arguments: Value,
    pub kind: RequestKind,
    #[pyo3(get)]
    pub key: Option<String>,
    pub data: Value,
    pub metadata: Value,
}

impl InputRequested {
    pub fn new(call: &ToolCall, request: &Request) -> Self {
        InputRequested {
            call_id: call.id.clone(),
            tool_name: call.name.clone(),
            arguments: call.arguments.clone(),
            kind: request.kind,
            key: request.key.clone(),
            data: request.data.clone(),
            metadata: request.metadata.clone(),
        }
    }
}

#[pymethods]
impl InputRequested {
    /// The tool call's arguments, as a dict.
    #[getter]
    fn arguments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.arguments)
    }

    /// `"approval"`, `"input"` (a `ctx.ask` question) or `"frontend"` (a frontend tool).
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind.name()
    }

    /// What the tool (or hook) wants to show the person.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.data)
    }

    /// Data for the app only; the model never sees it.
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "InputRequested(call_id={:?}, tool_name={:?}, kind={:?}, data={})",
            self.call_id,
            self.tool_name,
            self.kind(),
            self.data
        )
    }
}

/// The requests of the calls still waiting at the end of `messages`.
pub fn pending_requests(messages: &[Message]) -> Vec<InputRequested> {
    waiting_calls(messages)
        .into_iter()
        .filter_map(|(call, message)| {
            let Message::Tool { requests, .. } = message else {
                return None;
            };
            let request = requests.iter().find(|request| request.answer.is_none())?;
            Some(InputRequested::new(&call, request))
        })
        .collect()
}

/// The tool calls of the last assistant message that are still waiting, each
/// with its newest tool message. (A resumed call adds a new tool message for
/// the same call: the history only grows.)
pub fn waiting_calls(messages: &[Message]) -> Vec<(ToolCall, &Message)> {
    let Some(assistant_position) = messages
        .iter()
        .rposition(|message| matches!(message, Message::Assistant { .. }))
    else {
        return Vec::new();
    };
    let Message::Assistant { tool_calls, .. } = &messages[assistant_position] else {
        unreachable!("rposition found an assistant message");
    };
    let later_messages = &messages[assistant_position + 1..];
    tool_calls
        .iter()
        .filter_map(|call| {
            let newest = later_messages
                .iter()
                .rev()
                .find(|message| matches!(message, Message::Tool { tool_call_id, .. } if *tool_call_id == call.id))?;
            newest.is_waiting().then(|| (call.clone(), newest))
        })
        .collect()
}

/// Checks that `reply` fits the request: approvals take `Approve` or `Deny`,
/// questions and frontend tools take `Answer` or `Deny`.
pub fn check_reply(kind: RequestKind, reply: &Reply) -> PyResult<()> {
    let fits = match reply {
        Reply::Deny { .. } => true,
        Reply::Approve { .. } => kind == RequestKind::Approval,
        Reply::Answer { .. } => kind != RequestKind::Approval,
    };
    if fits {
        Ok(())
    } else {
        Err(PyTypeError::new_err(match kind {
            RequestKind::Approval => "Reply to an approval with Approve(...) or Deny(...)",
            _ => "Reply to a question or a frontend tool with Answer(...) or Deny(...)",
        }))
    }
}

/// The result the model gets for a call nobody answered (a new prompt came instead).
pub fn unanswered_note(waiting_message: &Message) -> String {
    let Message::Tool { requests, .. } = waiting_message else {
        return "The user did not answer".to_string();
    };
    match requests.iter().find(|request| request.answer.is_none()) {
        Some(request) if request.kind == RequestKind::Approval => {
            "The tool did not run: the user did not answer the approval request".to_string()
        }
        Some(request) if !request.data.is_null() => format!("The user did not answer: {}", request.data),
        _ => "The user did not answer".to_string(),
    }
}
