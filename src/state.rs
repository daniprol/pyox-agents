//! The run's state: one JSON document that tools read and change, shared
//! with the app (AG-UI shared state, a `data-state` part for the AI SDK).
//!
//! - The app gives it with `execute(state=...)` and gets it back in
//!   `RunFinished.state`. Nothing is stored: the app keeps it between runs.
//! - A tool reads and changes it through `ctx.state`, like a dict. When the
//!   tool finishes, its changes are applied to the run's state and sent as a
//!   `StateChanged` event (the new state and a JSON Patch of what changed).
//! - A tool that fails, or stops to wait for a person, changes nothing: it
//!   runs again from its start once answered.
//! - Tools running at the same time each apply only what they changed, so
//!   changes to different keys never overwrite each other.
//!
//! A subagent sees the state only if its tool says so (`as_tool(state="read" | "write")`).

use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyPermissionError;
use pyo3::prelude::*;
use serde_json::Value;

use crate::events::{AgentEvent, StateChanged};
use crate::human_input::SharedCallLog;
use crate::json_convert::python_to_json;
use crate::run_context::Context;

/// What an agent may do with the run's state.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StateAccess {
    /// No access: `ctx.state` raises (subagents, unless their tool allows it).
    None,
    Read,
    /// Read and change. The main agent always has it.
    Write,
}

impl StateAccess {
    /// Reads `as_tool(state=...)`: `None`, `"read"` or `"write"`.
    pub fn from_python(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if value.is_none() {
            return Ok(StateAccess::None);
        }
        match value.extract::<String>()?.as_str() {
            "read" => Ok(StateAccess::Read),
            "write" => Ok(StateAccess::Write),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown state={other:?}: use None, \"read\" or \"write\""
            ))),
        }
    }
}

/// The run's state as one agent sees it. Cloning it shares the same document.
#[derive(Clone)]
pub struct RunState {
    /// `Arc<Mutex<...>>`: shared by every tool call (and subagent) of the run.
    document: Arc<Mutex<Value>>,
    access: StateAccess,
}

impl RunState {
    /// The state of a new run: the main agent can read and change it.
    pub fn new(document: Value) -> Self {
        RunState {
            document: Arc::new(Mutex::new(document)),
            access: StateAccess::Write,
        }
    }

    /// The same state, as a subagent with `access` sees it.
    pub fn with_access(&self, access: StateAccess) -> Self {
        RunState {
            document: self.document.clone(),
            access,
        }
    }

    /// A copy of the state as it is now (even without access: for `RunFinished`).
    pub fn current(&self) -> Value {
        self.document.lock().unwrap().clone()
    }

    /// A copy of the state, for `ctx.state`; an error without access.
    pub fn read(&self) -> PyResult<Value> {
        if self.access == StateAccess::None {
            return Err(PyPermissionError::new_err(
                "This subagent cannot see the state: give its tool `state=\"read\"` or `state=\"write\"`",
            ));
        }
        Ok(self.current())
    }

    /// Applies what a tool changed (`before` is what it read, `after` what it
    /// left) to the state as it is now. Returns the event to send, or `None`
    /// when nothing changed; `Err` is the error result for the model.
    pub fn apply_changes(&self, before: &Value, after: &Value) -> Result<Option<StateChanged>, String> {
        let changes = json_patch::diff(before, after);
        if changes.is_empty() {
            return Ok(None);
        }
        if self.access != StateAccess::Write {
            return Err("This subagent can only read the state, not change it".to_string());
        }
        let mut document = self.document.lock().unwrap();
        // `patch` changes nothing if any operation fails.
        json_patch::patch(&mut document, &changes)
            .map_err(|error| format!("The state changed while the tool ran, and its changes do not fit: {error}"))?;
        Ok(Some(StateChanged {
            state: document.clone(),
            delta: serde_json::to_value(&changes).expect("a JSON Patch always serializes"),
            subagent_id: None, // set by the event sender
        }))
    }
}

/// After a tool finished: applies what it changed through `ctx.state` (if it
/// used it) and sends the `StateChanged` event. `Err` is the error result for the model.
pub async fn apply_state_changes(call_log: &SharedCallLog, context: &Context) -> Result<(), String> {
    let Some(view) = call_log.lock().unwrap().state_view.take() else {
        return Ok(()); // the tool did not touch the state
    };
    let after = Python::attach(|py| python_to_json(view.object.bind(py)))
        .map_err(|error| format!("The tool left a state that is not JSON: {error}"))?;
    if let Some(changed) = context.state.apply_changes(&view.before, &after)? {
        context.events().send(AgentEvent::StateChanged(changed)).await;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tools_changing_different_keys_at_the_same_time_keep_both_changes() {
        let state = RunState::new(json!({"todos": [], "title": "Trip"}));
        let read_by_both = state.current();

        let first = state.apply_changes(&read_by_both, &json!({"todos": ["pack"], "title": "Trip"}));
        let second = state.apply_changes(&read_by_both, &json!({"todos": [], "title": "Lisbon"}));

        assert!(first.unwrap().is_some());
        let second = second.unwrap().unwrap();
        assert_eq!(
            second.delta,
            json!([{"op": "replace", "path": "/title", "value": "Lisbon"}])
        );
        assert_eq!(state.current(), json!({"todos": ["pack"], "title": "Lisbon"}));
    }

    #[test]
    fn nothing_changed_sends_nothing_and_read_only_access_cannot_change() {
        let state = RunState::new(json!({"count": 1}));
        assert!(
            state
                .apply_changes(&json!({"count": 1}), &json!({"count": 1}))
                .unwrap()
                .is_none()
        );

        let reader = state.with_access(StateAccess::Read);
        assert!(
            reader
                .apply_changes(&json!({"count": 1}), &json!({"count": 2}))
                .is_err()
        );
        assert_eq!(state.current(), json!({"count": 1}));
    }
}
