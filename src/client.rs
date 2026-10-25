//! `Client`: everything needed to talk to the API.
//!
//! Create one `Client` and share it between agents: it keeps a pool of open
//! connections, so later requests skip the TCP/TLS handshake.

use std::collections::HashMap;
use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// The settings an agent run needs to send requests.
/// Cloning is cheap: `reqwest::Client` is a shared handle to one connection pool.
#[derive(Clone)]
pub struct ClientSettings {
    pub http_client: reqwest::Client,
    pub api_key: Option<String>,
    pub base_url: String,
    /// How many times to resend a request that failed before any response arrived.
    pub max_retries: u32,
}

impl ClientSettings {
    /// Sends `body` as JSON to `base_url + path`.
    ///
    /// If the request fails before any response arrives (typically a pooled
    /// connection the server had already closed), nothing was received yet,
    /// so it is safe to send it again. We retry up to `max_retries` times,
    /// waiting 0.1 s, then 0.2 s, 0.4 s... Once a response starts, we never
    /// retry: streamed text may already have reached the user.
    pub async fn post_json(&self, path: &str, body: &Value) -> Result<reqwest::Response, reqwest::Error> {
        let mut retries_done = 0;
        loop {
            let mut request = self.http_client.post(format!("{}{path}", self.base_url)).json(body);
            if let Some(api_key) = &self.api_key {
                request = request.bearer_auth(api_key);
            }
            match request.send().await {
                Err(error) if retries_done < self.max_retries && failed_before_response(&error) => {
                    tokio::time::sleep(Duration::from_millis(100 << retries_done)).await;
                    retries_done += 1;
                }
                result => return result,
            }
        }
    }
}

/// True for connection problems; false for timeouts (retrying would only make the wait longer).
fn failed_before_response(error: &reqwest::Error) -> bool {
    (error.is_connect() || error.is_request()) && !error.is_timeout()
}

/// Python: `Client(api_key=None, *, base_url=..., headers=None, ...)`.
///
/// `frozen` means the Python object cannot be modified after creation,
/// which makes it safe to share between concurrent runs.
#[pyclass(frozen, module = "pyox_agents")]
pub struct Client {
    pub settings: ClientSettings,
}

#[pymethods]
impl Client {
    /// `api_key` and `base_url` default to the `OPENAI_API_KEY` and
    /// `OPENAI_BASE_URL` environment variables.
    /// `request_timeout_seconds` covers the whole streamed response, so keep it generous.
    /// `max_retries`: how many times to resend a request that failed before any response arrived.
    #[new]
    #[pyo3(signature = (
        api_key = None,
        *,
        base_url = None,
        headers = None,
        connect_timeout_seconds = 10.0,
        request_timeout_seconds = 600.0,
        max_retries = 2,
    ))]
    fn new(
        api_key: Option<String>,
        base_url: Option<String>,
        headers: Option<HashMap<String, String>>,
        connect_timeout_seconds: f64,
        request_timeout_seconds: Option<f64>,
        max_retries: u32,
    ) -> PyResult<Self> {
        let mut default_headers = HeaderMap::new();
        for (name, value) in headers.unwrap_or_default() {
            let header_name = HeaderName::try_from(name.as_str())
                .map_err(|error| PyValueError::new_err(format!("Invalid header name {name:?}: {error}")))?;
            let header_value = HeaderValue::try_from(value.as_str())
                .map_err(|error| PyValueError::new_err(format!("Invalid value for header {name:?}: {error}")))?;
            default_headers.insert(header_name, header_value);
        }

        let mut builder = reqwest::Client::builder()
            .default_headers(default_headers)
            .connect_timeout(Duration::from_secs_f64(connect_timeout_seconds));
        if let Some(seconds) = request_timeout_seconds {
            builder = builder.timeout(Duration::from_secs_f64(seconds));
        }
        let http_client = builder
            .build()
            .map_err(|error| PyValueError::new_err(format!("Could not create HTTP client: {error}")))?;

        Ok(Client {
            settings: ClientSettings {
                http_client,
                api_key: api_key.or_else(|| std::env::var("OPENAI_API_KEY").ok()),
                base_url: base_url
                    .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
                    .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
                    .trim_end_matches('/')
                    .to_string(),
                max_retries,
            },
        })
    }

    #[getter]
    fn base_url(&self) -> &str {
        &self.settings.base_url
    }

    fn __repr__(&self) -> String {
        format!("Client(base_url={:?})", self.settings.base_url)
    }
}
