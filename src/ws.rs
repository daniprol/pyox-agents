//! One WebSocket connection to the Responses API.
//!
//! Over WebSocket, each model turn is one JSON message we send
//! (`{"type": "response.create", ...}`), answered by the same events the HTTP
//! stream would send (`response.output_text.delta`, ..., `response.completed`),
//! one per WebSocket message.
//!
//! A connection runs one response at a time. Azure requires it, and it keeps
//! cancelling simple: closing the connection stops the response.
//!
//! The connection is opened by our usual `reqwest` client: a `GET` to
//! `base_url + "/responses"` asking to switch ("upgrade") to WebSocket. This is
//! exactly what `wss://api.openai.com/v1/responses` means (`wss://` is
//! WebSocket over `https://`), and it reuses the client's TLS setup, proxy
//! settings, connect timeout and custom headers. Once the server agrees, the
//! connection is handed to `tokio-tungstenite`, which only reads and writes
//! WebSocket messages on it.

use std::time::Instant;

use futures::{SinkExt, StreamExt};
use pyo3::prelude::*;
use reqwest::header::{CONNECTION, SEC_WEBSOCKET_ACCEPT, SEC_WEBSOCKET_KEY, SEC_WEBSOCKET_VERSION, UPGRADE};
use serde_json::Value;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::handshake::{client::generate_key, derive_accept_key};
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::{self, Message};

use crate::client::{ClientSettings, failed_before_response};
use crate::errors::{InvalidResponseError, NetworkError, error_for_http_failure, error_for_http_status};

/// An open connection, ready for its next `response.create`.
pub struct ResponsesSocket {
    stream: WebSocketStream<reqwest::Upgraded>,
    pub opened_at: Instant,
    /// The newest response that finished on this connection during the
    /// current run. The server keeps recent responses in a cache that belongs
    /// to the connection, so the run's next turn can continue from this one
    /// (even without `store`), but only on this same connection.
    pub cached_response_id: Option<String>,
}

/// Why a connection could not be opened.
pub enum OpenError {
    /// The server answered 404, 405 or 426: it does not speak WebSocket here.
    /// The client then uses HTTP instead.
    NotSupported,
    /// The connection failed before the server answered: worth another try.
    Retryable(PyErr),
    /// Anything else (a wrong key, a rate limit, a timeout...): raised as is.
    Failed(PyErr),
}

impl ResponsesSocket {
    /// Opens a new connection with the client's address, key and headers.
    pub async fn open(client: &ClientSettings) -> Result<Self, OpenError> {
        // A random key the server must answer with a matching `Sec-WebSocket-Accept`,
        // which proves it really switched to WebSocket.
        let key = generate_key();
        let mut request = client
            .http_client
            .get(format!("{}/responses", client.base_url))
            .header(CONNECTION, "Upgrade")
            .header(UPGRADE, "websocket")
            .header(SEC_WEBSOCKET_VERSION, "13")
            .header(SEC_WEBSOCKET_KEY, &key);
        if let Some(api_key) = &client.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = match request.send().await {
            Ok(response) => response,
            Err(error) if failed_before_response(&error) => {
                return Err(OpenError::Retryable(error_for_http_failure(error)));
            }
            Err(error) => return Err(OpenError::Failed(error_for_http_failure(error))),
        };
        match response.status().as_u16() {
            101 => {} // "Switching Protocols": the server agreed
            404 | 405 | 426 => return Err(OpenError::NotSupported),
            status => {
                let body = response.text().await.unwrap_or_default();
                return Err(OpenError::Failed(error_for_http_status(status, &body)));
            }
        }
        let expected_accept = derive_accept_key(key.as_bytes());
        let accepted = response
            .headers()
            .get(SEC_WEBSOCKET_ACCEPT)
            .is_some_and(|accept| *accept == *expected_accept);
        if !accepted {
            return Err(OpenError::Failed(InvalidResponseError::new_err(
                "The server did not accept the WebSocket connection properly (wrong Sec-WebSocket-Accept)",
            )));
        }

        let connection = response
            .upgrade()
            .await
            .map_err(|error| OpenError::Retryable(NetworkError::new_err(error.to_string())))?;
        Ok(ResponsesSocket {
            stream: WebSocketStream::from_raw_socket(connection, Role::Client, None).await,
            opened_at: Instant::now(),
            cached_response_id: None,
        })
    }

    /// Sends one JSON message (a `response.create`).
    pub async fn send(&mut self, message: &Value) -> PyResult<()> {
        self.stream
            .send(Message::text(message.to_string()))
            .await
            .map_err(network_error)
    }

    /// Waits for the next event and returns its JSON text. `None` means the
    /// server closed the connection.
    ///
    /// tungstenite answers the server's pings by itself; we skip them here.
    pub async fn next_event(&mut self) -> Option<PyResult<String>> {
        loop {
            // `?` on an `Option`: if the stream has ended (`None`), return `None` right away.
            match self.stream.next().await? {
                Ok(Message::Text(text)) => return Some(Ok(text.as_str().to_string())),
                Ok(Message::Close(_)) => return None,
                Ok(_) => continue, // pings, pongs, binary messages: not events
                Err(error) => return Some(Err(network_error(error))),
            }
        }
    }
}

fn network_error(error: tungstenite::Error) -> PyErr {
    NetworkError::new_err(format!("WebSocket error: {error}"))
}
