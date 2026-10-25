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
//! └── MaxTurnsExceededError    the agent hit its `max_turns` limit
//! ```

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

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

/// Picks the right exception for an HTTP error status code.
pub fn error_for_http_status(status_code: u16, response_body: &str) -> PyErr {
    let message = format!("HTTP {status_code}: {response_body}");
    match status_code {
        401 | 403 => AuthenticationError::new_err(message),
        429 => RateLimitError::new_err(message),
        _ => ApiError::new_err(message),
    }
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
    module.add("AgentError", py.get_type::<AgentError>())?;
    module.add("ApiError", py.get_type::<ApiError>())?;
    module.add("AuthenticationError", py.get_type::<AuthenticationError>())?;
    module.add("RateLimitError", py.get_type::<RateLimitError>())?;
    module.add("RequestTimeoutError", py.get_type::<RequestTimeoutError>())?;
    module.add("NetworkError", py.get_type::<NetworkError>())?;
    module.add("InvalidResponseError", py.get_type::<InvalidResponseError>())?;
    module.add("MaxTurnsExceededError", py.get_type::<MaxTurnsExceededError>())?;
    Ok(())
}
