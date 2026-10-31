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
//! ├── MaxTurnsExceededError    the agent hit its `max_turns` limit
//! ├── HistoryError             a history store failed to load or save
//! ├── HookError                a Rust hook failed, or a hook returned something invalid
//! └── PendingRequestsError     a new prompt came while tool calls wait for answers
//! ```
//!
//! `StopToolCall` is not an error: `ctx.ask` raises it to stop a tool that
//! must wait (or was denied). It derives from `BaseException`, like
//! `asyncio.CancelledError`, so a tool's `except Exception:` does not catch it.

use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyBaseException, PyException};
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyox_tools::usage::Usage;
use reqwest::header::HeaderMap;

use crate::json_convert::json_to_python;

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
    StopToolCall,
    PyBaseException,
    "Raised by `ctx.ask` to stop a tool that must wait for an answer or was denied. Do not catch it."
);

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

/// Builds the `ApiError` (or subclass) for an error the API answered with.
/// Its details become attributes, so users never have to parse the message:
///
/// - `status_code`: the HTTP status (`None` for an error inside a stream)
/// - `body`: the error body, parsed as JSON when it is JSON (else the text)
/// - `request_id`: OpenAI's id for the request (`x-request-id`), for support
/// - `retry_after_seconds`: how long the server asked to wait, if it said
pub fn api_error(
    status_code: Option<u16>,
    body: &str,
    request_id: Option<String>,
    retry_after_seconds: Option<f64>,
) -> PyErr {
    let error = match status_code {
        Some(status @ (401 | 403)) => AuthenticationError::new_err(format!("HTTP {status}: {body}")),
        Some(429) => RateLimitError::new_err(format!("HTTP 429: {body}")),
        Some(status) => ApiError::new_err(format!("HTTP {status}: {body}")),
        None => ApiError::new_err(body.to_string()),
    };
    Python::attach(|py| {
        let value = error.value(py);
        let parsed_body = match serde_json::from_str::<serde_json::Value>(body) {
            Ok(json) => json_to_python(py, &json).map(Bound::unbind),
            Err(_) => Ok(PyString::new(py, body).into_any().unbind()),
        };
        // `let _ =` ignores a failure: the error itself matters more than its details.
        let _ = value.setattr("status_code", status_code);
        let _ = parsed_body.and_then(|parsed_body| value.setattr("body", parsed_body));
        let _ = value.setattr("request_id", request_id);
        let _ = value.setattr("retry_after_seconds", retry_after_seconds);
    });
    error
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
    if let Some(milliseconds) = header_text(headers, "retry-after-ms").and_then(|text| text.parse::<f64>().ok()) {
        return Some(Duration::from_secs_f64(milliseconds.max(0.0) / 1000.0));
    }
    let seconds = header_text(headers, "retry-after")?.parse::<f64>().ok()?;
    Some(Duration::from_secs_f64(seconds.max(0.0)))
}

/// Converts an error from the HTTP library (`reqwest`) into our exceptions.
pub fn error_for_http_failure(error: reqwest::Error) -> PyErr {
    if error.is_timeout() {
        RequestTimeoutError::new_err(error.to_string())
    } else {
        NetworkError::new_err(error.to_string())
    }
}

pub fn add_exceptions_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let agent_error = py.get_type::<AgentError>();
    // `None` unless the error was raised during a run (see `with_usage`).
    agent_error.setattr("usage", py.None())?;
    module.add("AgentError", agent_error)?;
    let api_error = py.get_type::<ApiError>();
    // `None` until an error sets them (see `api_error`).
    for attribute in ["status_code", "body", "request_id", "retry_after_seconds"] {
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
    module.add("StopToolCall", py.get_type::<StopToolCall>())?;
    Ok(())
}
