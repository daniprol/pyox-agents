//! The OpenAI Decisions API (`POST /decisions`): fast, cheap answers to
//! questions about an input, given as probabilities.
//!
//! It is not a model turn of an agent, so it does not go through
//! `model_api.rs`: Python calls `decide(...)` (see `pyox_agents.decisions`),
//! which builds the questions and calls `send_decision` here, with the same
//! `Client` the agents use.
//!
//! ```text
//! request:  {"model": "gpt-6-luna", "input": "The deploy failed...",
//!            "questions": [{"type": "predicate", "name": "urgent", "instructions": "Needs attention now?"}]}
//! response: {"model": "gpt-6-luna",
//!            "answers": [{"type": "predicate", "name": "urgent", "probability": 0.97}],
//!            "usage": {"input_tokens": 42, ...}}
//! ```
//!
//! The input is text, a prompt (text, images) or one of our histories. A
//! history becomes the same `input` items as for the Responses API, without
//! reasoning (the Decisions API does not take it).

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyox_tools::usage::Usage;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::client::Client;
use crate::errors::{InvalidResponseError, error_for_http_failure, error_for_http_status};
use crate::history_store::messages_from_python;
use crate::json_convert::{json_to_python, python_to_json};
use crate::messages::prompt_from_python;
use crate::openai_responses::{ResponseUsage, input_items, user_content};

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Sends one Decisions API request. Python awaits the returned `Decision`.
///
/// `questions` are already in the API's format (Python's `Predicate`,
/// `Choice` and `Score` build them). Optional fields are only sent when given.
#[pyfunction]
#[pyo3(signature = (client, input, questions, *, model, safety_identifier = None, prompt_cache_key = None))]
pub fn send_decision<'py>(
    py: Python<'py>,
    client: &Client,
    input: &Bound<'py, PyAny>,
    questions: &Bound<'py, PyAny>,
    model: String,
    safety_identifier: Option<String>,
    prompt_cache_key: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut request_body = Map::new();
    request_body.insert("model".into(), json!(model));
    request_body.insert("input".into(), decision_input(input)?);
    request_body.insert("questions".into(), python_to_json(questions)?);
    if let Some(safety_identifier) = safety_identifier {
        request_body.insert("safety_identifier".into(), json!(safety_identifier));
    }
    if let Some(prompt_cache_key) = prompt_cache_key {
        request_body.insert("prompt_cache_key".into(), json!(prompt_cache_key));
    }
    let request_body = Value::Object(request_body);

    // The client's settings are cheap to clone (shared handles). The request
    // runs on the Tokio runtime; `future_into_py` turns it into a Python awaitable.
    let client = client.settings.clone();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let response = client
            .post_json("/decisions", &request_body)
            .await
            .map_err(error_for_http_failure)?;
        let status = response.status();
        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let response_body = response.text().await.map_err(error_for_http_failure)?;
        if !status.is_success() {
            return Err(error_for_http_status(status.as_u16(), &response_body));
        }
        read_decision(&response_body, request_id)
    })
}

/// The request's `input`: text as is; a history as `input` items; a prompt
/// (a list of text, images...) as one user message.
fn decision_input(input: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(text) = input.extract::<String>() {
        return Ok(json!(text));
    }
    if is_history(input) {
        let messages = messages_from_python(input)?;
        return Ok(Value::Array(input_items(&messages, None)));
    }
    let prompt = prompt_from_python(input)?;
    Ok(json!([{"role": "user", "content": user_content(&prompt)}]))
}

/// A history is a list of message dicts, which all have a `"role"`. (Prompt
/// parts are strings, or dicts with a `"type"`.) Like Python's
/// `"role" in input[0]`, where any error means "no".
fn is_history(input: &Bound<'_, PyAny>) -> bool {
    input
        .get_item(0)
        .and_then(|first_item| first_item.contains("role"))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct DecisionResponse {
    model: String,
    answers: Vec<Answer>,
    #[serde(default)]
    usage: ResponseUsage,
}

/// One answer. `#[serde(tag = "type")]` picks the variant from the answer's
/// `"type"` field (`"predicate"`, `"choice"`, ...), and the other fields fill
/// the class inside it.
#[derive(Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Answer {
    Predicate(PredicateAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
    Refusal(Refusal),
}

impl Answer {
    fn name(&self) -> &str {
        match self {
            Answer::Predicate(answer) => &answer.name,
            Answer::Choice(answer) => &answer.name,
            Answer::Score(answer) => &answer.name,
            Answer::Refusal(answer) => &answer.name,
        }
    }

    /// The answer as its Python class (a copy: answers are small).
    fn to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Answer::Predicate(answer) => Py::new(py, answer.clone())?.into_any(),
            Answer::Choice(answer) => Py::new(py, answer.clone())?.into_any(),
            Answer::Score(answer) => Py::new(py, answer.clone())?.into_any(),
            Answer::Refusal(answer) => Py::new(py, answer.clone())?.into_any(),
        })
    }
}

fn read_decision(response_body: &str, request_id: Option<String>) -> PyResult<Decision> {
    let response: DecisionResponse = serde_json::from_str(response_body)
        .map_err(|error| InvalidResponseError::new_err(format!("{error}: {response_body}")))?;
    Ok(Decision {
        model: response.model,
        usage: response.usage.into_usage(),
        request_id,
        answers: response.answers,
    })
}

// ---------------------------------------------------------------------------
// The classes Python receives
// ---------------------------------------------------------------------------

/// The answers to one `decide(...)` call: `decision["urgent"]` is the answer
/// to the question named "urgent".
#[pyclass(frozen, module = "pyox_agents")]
pub struct Decision {
    /// The model that answered.
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    usage: Usage,
    /// OpenAI's id for this request (the `x-request-id` header), for support.
    #[pyo3(get)]
    request_id: Option<String>,
    /// In the order the questions were asked.
    answers: Vec<Answer>,
}

#[pymethods]
impl Decision {
    /// `decision[name]`: raises `KeyError` if no question had that name.
    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        match self.answers.iter().find(|answer| answer.name() == name) {
            Some(answer) => answer.to_python(py),
            None => Err(PyKeyError::new_err(name.to_string())),
        }
    }

    /// Every answer by question name: `{name: answer}`.
    #[getter]
    fn answers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let answers = PyDict::new(py);
        for answer in &self.answers {
            answers.set_item(answer.name(), answer.to_python(py)?)?;
        }
        Ok(answers)
    }

    fn __len__(&self) -> usize {
        self.answers.len()
    }

    fn __repr__(&self) -> String {
        let names: Vec<&str> = self.answers.iter().map(Answer::name).collect();
        format!("Decision(model={:?}, answers={names:?})", self.model)
    }
}

/// The answer to a `Predicate`: how likely the statement is true, from 0 to 1.
#[pyclass(frozen, get_all, module = "pyox_agents")]
#[derive(Clone, Deserialize)]
pub struct PredicateAnswer {
    #[serde(default)] // the API sends `null` for unnamed questions; ours always have names
    name: String,
    probability: f64,
}

/// The answer to a `Choice`: the most likely option, how sure the model is,
/// and the probability of each option.
#[pyclass(frozen, module = "pyox_agents")]
#[derive(Clone, Deserialize)]
pub struct ChoiceAnswer {
    #[pyo3(get)]
    #[serde(default)]
    name: String,
    /// A string or a boolean, like the options. Kept as JSON, read through the getter below.
    choice: Value,
    #[pyo3(get)]
    confidence: f64,
    probabilities: Vec<OptionProbability>,
}

#[derive(Clone, Deserialize)]
struct OptionProbability {
    value: Value,
    probability: f64,
}

/// The answer to a `Score`: `score` is the probability-weighted average of
/// the level indices (0 is the lowest level), so it can fall between levels.
#[pyclass(frozen, module = "pyox_agents")]
#[derive(Clone, Deserialize)]
pub struct ScoreAnswer {
    #[pyo3(get)]
    #[serde(default)]
    name: String,
    #[pyo3(get)]
    score: f64,
    #[pyo3(get)]
    confidence: f64,
    probabilities: Vec<LevelProbability>,
}

#[derive(Clone, Deserialize)]
struct LevelProbability {
    label: String,
    probability: f64,
}

/// The model declined to answer this question. The other answers are still valid.
#[pyclass(frozen, get_all, module = "pyox_agents")]
#[derive(Clone, Deserialize)]
pub struct Refusal {
    #[serde(default)]
    name: String,
}

#[pymethods]
impl PredicateAnswer {
    fn __repr__(&self) -> String {
        format!(
            "PredicateAnswer(name={:?}, probability={})",
            self.name, self.probability
        )
    }
}

#[pymethods]
impl ChoiceAnswer {
    #[getter]
    fn choice<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.choice)
    }

    /// `{option: probability}`, in the order of the options.
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let probabilities = PyDict::new(py);
        for option in &self.probabilities {
            probabilities.set_item(json_to_python(py, &option.value)?, option.probability)?;
        }
        Ok(probabilities)
    }

    fn __repr__(&self) -> String {
        format!(
            "ChoiceAnswer(name={:?}, choice={}, confidence={})",
            self.name, self.choice, self.confidence
        )
    }
}

#[pymethods]
impl ScoreAnswer {
    /// `{level label: probability}`, from the lowest level to the highest.
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let probabilities = PyDict::new(py);
        for level in &self.probabilities {
            probabilities.set_item(&level.label, level.probability)?;
        }
        Ok(probabilities)
    }

    fn __repr__(&self) -> String {
        format!(
            "ScoreAnswer(name={:?}, score={}, confidence={})",
            self.name, self.score, self.confidence
        )
    }
}

#[pymethods]
impl Refusal {
    fn __repr__(&self) -> String {
        format!("Refusal(name={:?})", self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_of_answer_is_read() {
        let response_body = json!({
            "model": "gpt-6-luna",
            "answers": [
                {"type": "predicate", "name": "urgent", "probability": 0.9},
                {"type": "choice", "name": "team", "choice": "infra", "confidence": 0.8,
                 "probabilities": [{"value": "infra", "probability": 0.8}, {"value": "billing", "probability": 0.2}]},
                {"type": "score", "name": "severity", "score": 1.5, "confidence": 0.6,
                 "probabilities": [{"value": 0, "label": "Low", "probability": 0.1},
                                   {"value": 1, "label": "High", "probability": 0.9}]},
                {"type": "refusal", "name": "secret"},
            ],
            "usage": {"input_tokens": 42, "input_tokens_details": {"cached_tokens": 2}, "output_tokens": 0},
        });

        let decision = read_decision(&response_body.to_string(), Some("req_1".into())).unwrap();

        let names: Vec<&str> = decision.answers.iter().map(Answer::name).collect();
        assert_eq!(names, ["urgent", "team", "severity", "secret"]);
        assert!(matches!(&decision.answers[0], Answer::Predicate(answer) if answer.probability == 0.9));
        assert!(matches!(&decision.answers[1], Answer::Choice(answer) if answer.choice == json!("infra")));
        assert!(matches!(&decision.answers[2], Answer::Score(answer) if answer.probabilities[1].label == "High"));
        assert!(matches!(&decision.answers[3], Answer::Refusal(_)));
        assert_eq!(
            (decision.usage.input_tokens, decision.usage.cached_input_tokens),
            (42, 2)
        );
        assert_eq!(decision.request_id.as_deref(), Some("req_1"));
    }

    #[test]
    fn a_response_of_another_shape_is_an_error() {
        assert!(read_decision(r#"{"answers": "nope"}"#, None).is_err());
    }
}
