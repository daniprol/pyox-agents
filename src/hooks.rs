//! Hooks (`Agent(hooks=[...])`): functions the agent loop calls at fixed
//! points of a run, to watch it or change what happens next.
//!
//! Each item of `hooks` is either:
//!
//! - **Python code**: any object with some of the hook methods
//!   (`before_tool_call(ctx, call)`, ...), sync or async. `pyox_agents.Hooks`
//!   builds one from plain functions.
//! - **Rust hooks** from a separately compiled package made with
//!   `pyox_tools::hooks`. The engine calls them directly, without Python
//!   (see `pyox_tools::native_hooks`).
//!
//! Hooks run in the order they were given. A hook that returns something
//! replaces the value it may change (see `Hook`), and the next hook sees
//! the new value. Returning `None` (Python) leaves it as it was.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyox_tools::hooks::ModelRequest;
use pyox_tools::native_hooks::NativeHooks;
use serde_json::Value;

pub use pyox_tools::native_hooks::Hook;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::errors::{HookError, wrap_user_error};
use crate::events::RunFinished;
use crate::json_convert::{json_to_python, python_to_json};
use crate::messages::{Message, ToolCall};
use crate::python_call::{PythonCalls, call_python};
use crate::run_context::Context;
use crate::tools::ToolResult;

/// All the hooks of an agent, in order.
#[derive(Default)]
pub struct AgentHooks {
    hook_sets: Vec<HookSet>,
    /// The `on_tool_error(ctx, call, error)` functions, in order. Python only:
    /// they get the exception object itself (see `agent::failure_result`).
    pub tool_error_handlers: Vec<Py<PyAny>>,
}

/// One item of `Agent(hooks=[...])`.
enum HookSet {
    /// The hook functions a Python object has, by hook.
    Python(Vec<(Hook, Py<PyAny>)>),
    /// Rust hooks, called directly. `owner` is the Python object that owns
    /// them: holding it keeps the hooks alive.
    Native {
        hooks: NativeHooks,
        #[allow(dead_code)] // never read, only held
        owner: Py<PyAny>,
    },
}

impl HookSet {
    fn has(&self, hook: Hook) -> bool {
        match self {
            HookSet::Python(functions) => functions.iter().any(|(name, _)| *name == hook),
            HookSet::Native { hooks, .. } => hooks.has(hook),
        }
    }
}

impl AgentHooks {
    pub fn from_python(items: &[Bound<'_, PyAny>]) -> PyResult<Self> {
        let mut hook_sets = Vec::new();
        let mut tool_error_handlers = Vec::new();
        for item in items {
            if let Some(native_hooks) = NativeHooks::from_python(item)? {
                hook_sets.push(HookSet::Native {
                    hooks: native_hooks,
                    owner: item.clone().unbind(),
                });
                continue;
            }
            // `getattr` with a default: `Hooks(...)` leaves unused hooks as `None`.
            let function_named = |name: &str| item.getattr(name).ok().filter(|function| !function.is_none());
            let mut functions = Vec::new();
            for hook in Hook::ALL {
                if let Some(function) = function_named(hook.name()) {
                    functions.push((hook, function.unbind()));
                }
            }
            let on_tool_error = function_named("on_tool_error");
            if functions.is_empty() && on_tool_error.is_none() {
                return Err(PyTypeError::new_err(format!(
                    "{} has no hooks: give an object with methods like `before_tool_call`, or use Hooks(...)",
                    item.repr()?
                )));
            }
            tool_error_handlers.extend(on_tool_error.map(Bound::unbind));
            hook_sets.push(HookSet::Python(functions));
        }
        Ok(AgentHooks {
            hook_sets,
            tool_error_handlers,
        })
    }

    // One method per hook, for the agent loop. Each one returns at once when
    // no hook set has that hook, so agents without hooks pay nothing.

    pub async fn before_run(&self, context: &Context, python_calls: &PythonCalls) -> PyResult<()> {
        if self.has(Hook::BeforeRun) {
            let ran = self.run(Hook::BeforeRun, context, Vec::new(), python_calls).await;
            ran.map_err(|error| wrap_user_error::<HookError>(error, "The before_run hook"))?;
        }
        Ok(())
    }

    /// The request as the hooks left it, or `None` when the agent has no
    /// `before_model_request` hooks. `make_request` builds the request only
    /// when there are hooks, so agents without them pay nothing.
    pub async fn before_model_request(
        &self,
        context: &Context,
        make_request: impl FnOnce() -> ModelRequest,
        python_calls: &PythonCalls,
    ) -> PyResult<Option<ModelRequest>> {
        if !self.has(Hook::BeforeModelRequest) {
            return Ok(None);
        }
        let arguments = vec![to_json(&make_request())];
        let changed = self
            .run(Hook::BeforeModelRequest, context, arguments, python_calls)
            .await;
        let changed = changed.map_err(|error| wrap_user_error::<HookError>(error, "The before_model_request hook"))?;
        from_json(Hook::BeforeModelRequest, changed).map(Some)
    }

    pub async fn after_model_response(
        &self,
        context: &Context,
        message: Message,
        python_calls: &PythonCalls,
    ) -> PyResult<Message> {
        if !self.has(Hook::AfterModelResponse) {
            return Ok(message);
        }
        let arguments = vec![to_json(&message)];
        let changed = self
            .run(Hook::AfterModelResponse, context, arguments, python_calls)
            .await;
        let changed = changed.map_err(|error| wrap_user_error::<HookError>(error, "The after_model_response hook"))?;
        match from_json(Hook::AfterModelResponse, changed)? {
            message @ Message::Assistant { .. } => Ok(message),
            _ => Err(HookError::new_err(
                "after_model_response must return an assistant message",
            )),
        }
    }

    pub async fn before_tool_call(
        &self,
        context: &Context,
        call: ToolCall,
        python_calls: &PythonCalls,
    ) -> PyResult<ToolCall> {
        if !self.has(Hook::BeforeToolCall) {
            return Ok(call);
        }
        let arguments = vec![to_json(&call)];
        let changed = self.run(Hook::BeforeToolCall, context, arguments, python_calls);
        from_json(Hook::BeforeToolCall, changed.await?)
    }

    pub async fn after_tool_call(
        &self,
        context: &Context,
        call: &ToolCall,
        result: ToolResult,
        python_calls: &PythonCalls,
    ) -> PyResult<ToolResult> {
        if !self.has(Hook::AfterToolCall) {
            return Ok(result);
        }
        let arguments = vec![to_json(call), to_json(&result)];
        let changed = self.run(Hook::AfterToolCall, context, arguments, python_calls);
        from_json(Hook::AfterToolCall, changed.await?)
    }

    /// Python hooks get the `RunFinished` event; Rust hooks get the whole conversation.
    pub async fn after_run(
        &self,
        context: &Context,
        finished: &RunFinished,
        python_calls: &PythonCalls,
    ) -> PyResult<()> {
        for hook_set in self.hook_sets.iter().filter(|hook_set| hook_set.has(Hook::AfterRun)) {
            match hook_set {
                HookSet::Python(functions) => {
                    let function = python_function(functions, Hook::AfterRun);
                    let finished = finished.clone();
                    call_python_hook(function, context, python_calls, move |py| {
                        Ok(vec![Py::new(py, finished)?.into_bound(py).into_any()])
                    })
                    .await
                    .map_err(|error| wrap_user_error::<HookError>(error, "The after_run hook"))?;
                }
                HookSet::Native { hooks, .. } => {
                    let arguments = Value::Array(vec![to_json(&finished.messages)]);
                    call_native_hook(*hooks, Hook::AfterRun, context, arguments).await?;
                }
            }
        }
        Ok(())
    }

    /// Whether any hook set has `hook`.
    fn has(&self, hook: Hook) -> bool {
        self.hook_sets.iter().any(|hook_set| hook_set.has(hook))
    }

    /// Runs every `hook` in order with `arguments` (what a Python hook gets
    /// after `ctx`). A hook may replace the last argument, so the next hook
    /// sees its change. Returns the last argument as it is at the end.
    async fn run(
        &self,
        hook: Hook,
        context: &Context,
        mut arguments: Vec<Value>,
        python_calls: &PythonCalls,
    ) -> PyResult<Option<Value>> {
        for hook_set in self.hook_sets.iter().filter(|hook_set| hook_set.has(hook)) {
            let replacement = match hook_set {
                HookSet::Python(functions) => {
                    let function = python_function(functions, hook);
                    let python_arguments = arguments.clone();
                    let result = call_python_hook(function, context, python_calls, move |py| {
                        python_arguments
                            .iter()
                            .map(|argument| json_to_python(py, argument))
                            .collect()
                    })
                    .await?;
                    // `None` means "no change".
                    Python::attach(|py| {
                        let result = result.bind(py);
                        if result.is_none() {
                            Ok(None)
                        } else {
                            python_to_json(result).map(Some)
                        }
                    })?
                }
                HookSet::Native { hooks, .. } => {
                    let changed_json = call_native_hook(*hooks, hook, context, Value::Array(arguments.clone())).await?;
                    // Empty text means "no change".
                    if changed_json.is_empty() {
                        None
                    } else {
                        let changed = serde_json::from_str(&changed_json).map_err(|error| {
                            HookError::new_err(format!("Rust hook {} returned invalid JSON: {error}", hook.name()))
                        })?;
                        Some(changed)
                    }
                }
            };
            if let (Some(replacement), Some(last_argument)) = (replacement, arguments.last_mut()) {
                *last_argument = replacement;
            }
        }
        Ok(arguments.pop())
    }
}

fn python_function(functions: &[(Hook, Py<PyAny>)], hook: Hook) -> Py<PyAny> {
    let (_, function) = functions
        .iter()
        .find(|(name, _)| *name == hook)
        .expect("only called for hook sets that have the hook");
    Python::attach(|py| function.clone_ref(py))
}

/// Calls `function(ctx, *arguments)`, awaiting it if it is async.
pub async fn call_python_hook(
    function: Py<PyAny>,
    context: &Context,
    python_calls: &PythonCalls,
    make_arguments: impl for<'py> FnOnce(Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> + Send + 'static,
) -> PyResult<Py<PyAny>> {
    let (context, ctx_calls) = (context.clone(), python_calls.clone());
    call_python(
        function,
        move |py| {
            let mut arguments = vec![context.to_python(py, None, ctx_calls)?.into_bound(py).into_any()];
            arguments.extend(make_arguments(py)?);
            Ok(arguments)
        },
        python_calls,
    )
    .await
}

/// Calls a Rust hook on a thread meant for blocking work, like Rust tools:
/// a hook may do slow work, and must not stall the network tasks.
async fn call_native_hook(hooks: NativeHooks, hook: Hook, context: &Context, arguments: Value) -> PyResult<String> {
    let context_json = serde_json::to_string(&context.run).expect("the context always serializes");
    let arguments_json = arguments.to_string();
    tokio::task::spawn_blocking(move || hooks.call(hook, &context_json, &arguments_json))
        .await
        .map_err(|_| HookError::new_err(format!("Rust hook {} crashed", hook.name())))?
        .map_err(HookError::new_err)
}

fn to_json(value: &impl Serialize) -> Value {
    serde_json::to_value(value).expect("hook arguments always serialize")
}

/// Reads back what the hooks returned. `run` gives back the last argument,
/// which every hook that calls this has.
fn from_json<Payload: DeserializeOwned>(hook: Hook, changed: Option<Value>) -> PyResult<Payload> {
    let changed = changed.expect("hooks that change something always have an argument");
    serde_json::from_value(changed)
        .map_err(|error| HookError::new_err(format!("A {} hook returned an invalid value: {error}", hook.name())))
}
