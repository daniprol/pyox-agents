//! Converts JSON values (`serde_json::Value`) to Python objects and back.
//!
//! We build Python objects directly (no JSON text in between), which avoids
//! printing JSON in Rust and parsing it again in Python.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value};

/// JSON -> Python: object -> dict, array -> list, string -> str, and so on.
pub fn json_to_python<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    let python_object = match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(boolean) => PyBool::new(py, *boolean).to_owned().into_any(),
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                integer.into_pyobject(py)?.into_any()
            } else if let Some(unsigned_integer) = number.as_u64() {
                unsigned_integer.into_pyobject(py)?.into_any()
            } else {
                number.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any()
            }
        }
        Value::String(text) => PyString::new(py, text).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_python(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(fields) => {
            let dict = PyDict::new(py);
            for (key, field_value) in fields {
                dict.set_item(key, json_to_python(py, field_value)?)?;
            }
            dict.into_any()
        }
    };
    Ok(python_object)
}

/// Python -> JSON. Accepts what `json.dumps` accepts (dict keys must be str).
pub fn python_to_json(object: &Bound<'_, PyAny>) -> PyResult<Value> {
    // `bool` is checked before `int` because in Python `True` is also an `int`.
    if object.is_none() {
        Ok(Value::Null)
    } else if let Ok(boolean) = object.cast::<PyBool>() {
        Ok(Value::Bool(boolean.is_true()))
    } else if object.is_instance_of::<PyInt>() {
        if let Ok(integer) = object.extract::<i64>() {
            Ok(Value::from(integer))
        } else {
            Ok(Value::from(object.extract::<u64>()?))
        }
    } else if object.is_instance_of::<PyFloat>() {
        let float: f64 = object.extract()?;
        Number::from_f64(float)
            .map(Value::Number)
            .ok_or_else(|| PyTypeError::new_err(format!("{float} cannot be represented in JSON")))
    } else if let Ok(text) = object.cast::<PyString>() {
        Ok(Value::String(text.to_str()?.to_owned()))
    } else if let Ok(dict) = object.cast::<PyDict>() {
        let mut fields = Map::with_capacity(dict.len());
        for (key, field_value) in dict.iter() {
            let key: String = key
                .extract()
                .map_err(|_| PyTypeError::new_err("JSON object keys must be strings"))?;
            fields.insert(key, python_to_json(&field_value)?);
        }
        Ok(Value::Object(fields))
    } else if object.is_instance_of::<PyList>() || object.is_instance_of::<PyTuple>() {
        let items = object
            .try_iter()?
            .map(|item| python_to_json(&item?))
            .collect::<PyResult<Vec<Value>>>()?;
        Ok(Value::Array(items))
    } else {
        Err(PyTypeError::new_err(format!(
            "Object of type {} is not JSON serializable",
            object.get_type().name()?
        )))
    }
}
