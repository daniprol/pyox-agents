//! The Rust core of pyox-agents.
//!
//! Python only defines the agent (prompt, tools) and iterates over events.
//! Everything that runs often or is heavy lives here in Rust:
//! HTTP requests, streaming (SSE) decoding, JSON parsing and the agent loop.
//!
//! Map of the modules (each file is small and self-contained):
//!
//! | Module          | What it does                                                  |
//! |-----------------|---------------------------------------------------------------|
//! | `client`        | `Client`: API key, base URL, headers, timeouts, HTTP pool     |
//! | `agent`         | `Engine` and `Run`: the agent loop and the event stream       |
//! | `events`        | The event classes Python receives (`TextDelta`, ...)          |
//! | `messages`      | Our own conversation history format                           |
//! | `openai_chat`   | Adapter for the OpenAI Chat Completions API                   |
//! | `sse`           | Splits a byte stream into server-sent events                  |
//! | `text_chunks`   | Groups streamed text (deltas, paragraphs, whole messages)     |
//! | `tools`         | Tool definitions and how to call Python or Rust tools         |
//! | `errors`        | The Python exception classes                                  |
//! | `json_convert`  | Converts JSON values to Python objects and back               |

use pyo3::prelude::*;

mod agent;
mod client;
mod errors;
mod events;
mod json_convert;
mod messages;
mod openai_chat;
mod sse;
mod text_chunks;
mod tools;

/// The compiled Python module, importable as `pyox_agents._core`.
/// The name must match `lib.name` in `Cargo.toml`.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<client::Client>()?;
    module.add_class::<agent::Engine>()?;
    module.add_class::<agent::Run>()?;
    module.add_class::<tools::BuiltinTool>()?;

    module.add_class::<events::TextDelta>()?;
    module.add_class::<events::ToolStarted>()?;
    module.add_class::<events::ToolFinished>()?;
    module.add_class::<events::RunFinished>()?;

    errors::add_exceptions_to_module(module)?;
    Ok(())
}
