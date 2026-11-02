//! The exceptions users can catch in Python.
//!
//! Inside Rust, a failing function returns `Err(PyErr)`. PyO3 turns that
//! into a raised Python exception when it crosses into Python.
//!
//! Hierarchy (all importable from `pyox_agents`):
//!
//! ```text
//! AgentError
//! ├── ApiError                 the API answered with an error
//! │   ├── AuthenticationError  401 / 403
//! │   └── RateLimitError       429
//! ├── RequestTimeoutError      the request took too long
//! ├── NetworkError             could not connect / connection dropped
//! ├── InvalidResponseError     the API sent something we could not understand
//! ├── MaxTurnsExceededError    the agent hit its `max_turns` limit (`error.messages` has the work so far)
//! ├── HistoryError             a history store failed (or raised) while loading or saving
//! ├── HookError                a hook or `on_request` raised, or returned something invalid
//! ├── PendingRequestsError     a new prompt came while tool calls wait for answers
//! └── StopRun                  a tool or hook ended the run on purpose
//! ```
//!
//! Everything a run raises is an `AgentError`, with the conversation so far
//! (`error.messages`) and the tokens used (`error.usage`). When your own code
//! (a hook, a history store, `on_request`) raised, the `HookError` or
//! `HistoryError` has your exception as `__cause__`. Mistakes in how pyox is
//! called (bad arguments to `Agent(...)` or `stream(...)`) stay plain
//! `ValueError` / `TypeError`.
//!
//! Tool errors are not raised: the model gets them as the tool's result (see
//! `tool_call::failure_result`). `ToolFailed` is the exception a tool raises
//! to choose that text.
//!
//! `StopToolCall` is not an error: `ctx.ask` raises it to stop a tool that
//! must wait (or was denied). It derives from `BaseException`, like
//! `asyncio.CancelledError`, so a tool's `except Exception:` does not catch it.

use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyBaseException, PyException};
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::PyString;
use pyox_tools::usage::Usage;
use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::events::messages_to_python;
use crate::json_convert::json_to_python;
use crate::messages::Message;

// Defined in Python, where they take extra arguments and have docstrings
// users read; Rust uses them through these types.
pyo3::import_exception!(pyox_agents.tools, ToolFailed);
pyo3::import_exception!(pyox_agents.tools, InvalidArguments);
pyo3::import_exception!(pyox_agents.human_input, ApprovalRequired);

create_exception!(
    pyox_agents,
    AgentError,
    PyException,
    "Base class for all pyox-agents errors."
);
create_exception!(pyox_agents, ApiError, AgentError, "The model API returned an error.");
create_exception!(
    pyox_agents,
    AuthenticationError,
    ApiError,
    "The API key was rejected (401/403)."
);
create_exception!(pyox_agents, RateLimitError, ApiError, "Too many requests (429).");
create_exception!(
    pyox_agents,
    RequestTimeoutError,
    AgentError,
    "The request took too long."
);
create_exception!(pyox_agents, NetworkError, AgentError, "Could not reach the API.");
create_exception!(
    pyox_agents,
    InvalidResponseError,
    AgentError,
    "The API response could not be parsed."
);
create_exception!(
    pyox_agents,
    MaxTurnsExceededError,
    AgentError,
    "The agent reached `max_turns`."
);

create_exception!(
    pyox_agents,
    HistoryError,
    AgentError,
    "A history store failed to load or save a conversation."
);
create_exception!(
    pyox_agents,
    HookError,
    AgentError,
    "A Rust hook failed, or a hook returned something invalid."
);

create_exception!(
    pyox_agents,
    PendingRequestsError,
    AgentError,
    "A new prompt came while tool calls wait for answers (with `Agent(unanswered=\"raise\")`)."
);
create_exception!(
    pyox_agents,
    StopRun,
    AgentError,
    "Raise in a tool or a hook to end the run. It is raised by the run, with `messages` and `usage`."
);
create_exception!(
    pyox_agents,
    StopToolCall,
    PyBaseException,
    "Raised by `ctx.ask` to stop a tool that must wait for an answer or was denied. Do not catch it."
);

/// Your code (a hook, a history store, `on_request`) raised `error`. pyox's
/// own errors (`StopRun`, or an `ApiError` your code let through) and
/// non-`Exception`s (`KeyboardInterrupt`) pass through as they are; anything
/// else becomes `Wrapper` ("The before_run hook raised ValueError: ..."),
/// with the original as `__cause__`, so `except AgentError` catches every
/// failure of a run and the original is one attribute away.
///
/// (`<Wrapper: PyTypeInfo>` means: any exception class made with `create_exception!`.)
pub fn wrap_user_error<Wrapper: PyTypeInfo>(error: PyErr, what: &str) -> PyErr {
    Python::attach(|py| {
        let passes_through = error.is_instance_of::<AgentError>(py) || !error.is_instance_of::<PyException>(py);
        if passes_through {
            return error;
        }
        let wrapped = PyErr::from_type(py.get_type::<Wrapper>(), format!("{what} raised {error}"));
        wrapped.set_cause(py, Some(error));
        wrapped
    })
}

/// An `AgentError` raised during a run gets the tokens the run used before it
/// failed, as `error.usage`: those tokens are billed even if the run failed.
pub fn with_usage(error: PyErr, usage: Usage) -> PyErr {
    Python::attach(|py| {
        if error.is_instance_of::<AgentError>(py) {
            // `let _ =` ignores a failure: the error itself matters more than its usage.
            let _ = error.value(py).setattr("usage", usage);
        }
    });
    error
}

/// An `AgentError` raised during a run also gets the conversation so far, as
/// `error.messages`, so the work is not lost: continue with `history=error.messages`.
pub fn with_messages(error: PyErr, messages: &[Message]) -> PyErr {
    Python::attach(|py| {
        if error.is_instance_of::<AgentError>(py)
            && let Ok(messages) = messages_to_python(py, messages)
        {
            let _ = error.value(py).setattr("messages", messages);
        }
    });
    error
}

/// Builds the `ApiError` (or subclass) for an error the API answered with.
/// Its details become attributes, so users never have to parse the message:
///
/// - `status_code`: the HTTP status (`None` for an error inside a stream)
/// - `code`: OpenAI's error code, like `"rate_limit_exceeded"`, if it gave one
/// - `body`: the error body, parsed as JSON when it is JSON (else the text)
/// - `request_id`: OpenAI's id for the request (`x-request-id`), for support
/// - `retry_after_seconds`: how long the server asked to wait, if it said
///
/// The message is short: the status and OpenAI's own message (the whole body
/// is in `body`).
pub fn api_error(
    status_code: Option<u16>,
    body: &str,
    request_id: Option<String>,
    retry_after_seconds: Option<f64>,
) -> PyErr {
    let parsed_body = serde_json::from_str::<Value>(body).ok();
    let details = parsed_body.as_ref().map(error_details).unwrap_or_default();
    let text = details.message.as_deref().unwrap_or(body);
    let message = match status_code {
        Some(status) => format!("HTTP {status}: {text}"),
        None => text.to_string(),
    };
    let error = match (status_code, details.code.as_deref()) {
        (Some(401 | 403), _) => AuthenticationError::new_err(message),
        // Inside a stream there is no status, only the code.
        (Some(429), _) | (None, Some("rate_limit_exceeded")) => RateLimitError::new_err(message),
        _ => ApiError::new_err(message),
    };
    Python::attach(|py| {
        let value = error.value(py);
        let body_for_python = match &parsed_body {
            Some(json) => json_to_python(py, json).map(Bound::unbind),
            None => Ok(PyString::new(py, body).into_any().unbind()),
        };
        // `let _ =` ignores a failure: the error itself matters more than its details.
        let _ = value.setattr("status_code", status_code);
        let _ = value.setattr("code", details.code);
        let _ = body_for_python.and_then(|body| value.setattr("body", body));
        let _ = value.setattr("request_id", request_id);
        let _ = value.setattr("retry_after_seconds", retry_after_seconds);
    });
    error
}

/// OpenAI's message and code for an error, from its JSON body.
#[derive(Default)]
struct ErrorDetails {
    message: Option<String>,
    code: Option<String>,
}

/// Finds the error object in the shapes OpenAI uses: `{"error": {...}}` (HTTP),
/// `{"response": {"error": {...}}}` (`response.failed`), or the event itself
/// (an `error` event, a Chat Completions stream error).
fn error_details(body: &Value) -> ErrorDetails {
    let error = body
        .get("error")
        .or_else(|| body.get("response").and_then(|response| response.get("error")))
        .filter(|error| error.is_object())
        .unwrap_or(body);
    let text = |key: &str| error.get(key).and_then(Value::as_str).map(str::to_string);
    ErrorDetails {
        message: text("message"),
        code: text("code"),
    }
}

/// Gives an `ApiError` raised while reading a stream the request's id, which
/// the stream itself does not repeat.
pub fn with_request_id(error: PyErr, request_id: &Option<String>) -> PyErr {
    Python::attach(|py| {
        let value = error.value(py);
        let has_no_id = value.getattr("request_id").is_ok_and(|id| id.is_none());
        if error.is_instance_of::<ApiError>(py) && has_no_id {
            let _ = value.setattr("request_id", request_id.clone());
        }
    });
    error
}

/// The `code` of an `ApiError` (like `"previous_response_not_found"`), if it has one.
pub fn error_code(error: &PyErr) -> Option<String> {
    Python::attach(|py| error.value(py).getattr("code").ok()?.extract().ok())
}

/// The API sent something we could not read. The raw text is cut to keep the
/// message readable.
pub fn invalid_response(problem: impl std::fmt::Display, raw_text: &str) -> PyErr {
    const MAX_CHARACTERS: usize = 200;
    let mut shown: String = raw_text.chars().take(MAX_CHARACTERS).collect();
    if shown.len() < raw_text.len() {
        shown.push_str("...");
    }
    InvalidResponseError::new_err(format!("{problem}: {shown}"))
}

/// The exception for an HTTP error status, from the response itself.
pub async fn error_for_response(response: reqwest::Response) -> PyErr {
    let status_code = response.status().as_u16();
    let request_id = header_text(response.headers(), "x-request-id");
    let retry_after_seconds = retry_after(response.headers()).map(|delay| delay.as_secs_f64());
    let body = response.text().await.unwrap_or_default();
    api_error(Some(status_code), &body, request_id, retry_after_seconds)
}

/// The exception for an HTTP error status, when only the status and body are known.
pub fn error_for_http_status(status_code: u16, response_body: &str) -> PyErr {
    api_error(Some(status_code), response_body, None, None)
}

/// A header as text, if the response has it.
pub fn header_text(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get(name)?.to_str().ok().map(str::to_string)
}

/// How long the server asks us to wait before sending the request again:
/// OpenAI's `retry-after-ms` (milliseconds), or the standard `retry-after`
/// (seconds). `None` if it does not say (or says it as a date, which OpenAI never does).
pub fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let number = |name: &str| {
        let number = header_text(headers, name)?.parse::<f64>().ok()?;
        // "inf" and "NaN" parse as numbers too, but are not a delay.
        number.is_finite().then_some(number.max(0.0))
    };
    if let Some(milliseconds) = number("retry-after-ms") {
        return Some(Duration::from_secs_f64(milliseconds / 1000.0));
    }
    // `try_from` fails for a delay too long to store, instead of crashing.
    Duration::try_from_secs_f64(number("retry-after")?).ok()
}

/// Converts an error from the HTTP library (`reqwest`) into our exceptions.
/// The message includes the underlying causes ("error sending request: ...:
/// connection refused"), which say what actually went wrong.
pub fn error_for_http_failure(error: reqwest::Error) -> PyErr {
    let mut message = error.to_string();
    // `source()` is the error's cause, like `__cause__` in Python.
    let mut cause = std::error::Error::source(&error);
    while let Some(inner) = cause {
        message.push_str(&format!(": {inner}"));
        cause = inner.source();
    }
    if error.is_timeout() {
        RequestTimeoutError::new_err(message)
    } else {
        NetworkError::new_err(message)
    }
}

pub fn add_exceptions_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let agent_error = py.get_type::<AgentError>();
    // `None` unless the error was raised during a run (see `with_usage`).
    agent_error.setattr("usage", py.None())?;
    agent_error.setattr("messages", py.None())?; // see `with_messages`
    module.add("AgentError", agent_error)?;
    let api_error = py.get_type::<ApiError>();
    // `None` until an error sets them (see `api_error`).
    for attribute in ["status_code", "code", "body", "request_id", "retry_after_seconds"] {
        api_error.setattr(attribute, py.None())?;
    }
    module.add("ApiError", api_error)?;
    module.add("AuthenticationError", py.get_type::<AuthenticationError>())?;
    module.add("RateLimitError", py.get_type::<RateLimitError>())?;
    module.add("RequestTimeoutError", py.get_type::<RequestTimeoutError>())?;
    module.add("NetworkError", py.get_type::<NetworkError>())?;
    module.add("InvalidResponseError", py.get_type::<InvalidResponseError>())?;
    module.add("MaxTurnsExceededError", py.get_type::<MaxTurnsExceededError>())?;
    module.add("HistoryError", py.get_type::<HistoryError>())?;
    module.add("HookError", py.get_type::<HookError>())?;
    module.add("PendingRequestsError", py.get_type::<PendingRequestsError>())?;
    module.add("StopRun", py.get_type::<StopRun>())?;
    module.add("StopToolCall", py.get_type::<StopToolCall>())?;
    Ok(())
}
