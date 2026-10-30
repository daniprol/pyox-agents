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
//! | `ws`            | One WebSocket connection to the Responses API                 |
//! | `ws_pool`       | Idle WebSocket connections shared by a client's runs          |
//! | `agent`         | `Engine`: the agent loop                                      |
//! | `run`           | `Run`: reading a run's events (one by one, batches, protocols) |
//! | `ai_sdk`        | A run as a Vercel AI SDK UI message stream (`useChat`)        |
//! | `ag_ui`         | A run as an AG-UI event stream                                |
//! | `events`        | The event classes Python receives (`TextDelta`, ...)          |
//! | `messages`      | Our own conversation history format                           |
//! | `history_store` | Reading given history; loading/saving it with history stores  |
//! | `hooks`         | Calling Python or Rust hooks at fixed points of a run         |
//! | `human_input`   | Tool calls that wait for a person: approvals, `ctx.ask`       |
//! | `run_context`   | `RunContext`: the `ctx` given to tools and hooks              |
//! | `model_api`     | Picks the API adapter; what a model request and answer hold   |
//! | `openai_chat`   | Adapter for the OpenAI Chat Completions API                   |
//! | `openai_responses` | Adapter for the OpenAI Responses API                       |
//! | `sse`           | Splits a byte stream into server-sent events                  |
//! | `text_chunks`   | Groups streamed text (deltas, paragraphs, whole messages)     |
//! | `tools`         | Tool definitions and how to call Python or Rust tools         |
//! | `errors`        | The Python exception classes                                  |
//! | `json_convert`  | Converts JSON values to Python objects and back               |
//! | `python_call`   | Calls Python functions (sync or async) from the agent loop    |

use pyo3::prelude::*;

mod ag_ui;
mod agent;
mod ai_sdk;
mod client;
mod errors;
mod events;
mod history_store;
mod hooks;
mod human_input;
mod json_convert;
mod messages;
mod model_api;
mod openai_chat;
mod openai_responses;
mod python_call;
mod run;
mod run_context;
mod sse;
mod text_chunks;
mod tools;
mod ws;
mod ws_pool;

/// The compiled Python module, importable as `pyox_agents._core`.
/// The name must match `lib.name` in `Cargo.toml`.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<client::Client>()?;
    module.add_class::<agent::Engine>()?;
    module.add_class::<run::Run>()?;
    module.add_class::<run::EventBatches>()?;
    module.add_class::<run::ProtocolStream>()?;
    module.add_class::<tools::BuiltinTool>()?;
    module.add_class::<run_context::PythonRunContext>()?;

    module.add_class::<events::TextDelta>()?;
    module.add_class::<events::ToolStarted>()?;
    module.add_class::<events::ToolFinished>()?;
    module.add_class::<events::RunFinished>()?;
    module.add_class::<human_input::InputRequested>()?;
    module.add_class::<pyox_tools::usage::Usage>()?;

    errors::add_exceptions_to_module(module)?;
    Ok(())
}
