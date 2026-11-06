//! Reads `SKILL.md` files: a YAML "frontmatter" block between `---` lines,
//! then the skill's instructions in Markdown.
//!
//! ```text
//! ---
//! name: pdf-processing
//! description: Extract text from PDFs. Use when handling PDFs.
//! ---
//! # PDF processing
//! ...
//! ```
//!
//! Python's standard library has no YAML parser, so this lives in Rust and the
//! Python package keeps zero dependencies. The rest of skills is Python
//! (`pyox_agents/skills.py`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Number, Value};
use yaml_rust2::{Yaml, YamlLoader};

use crate::json_convert::json_to_python;

/// `parse_skill_md(text) -> (frontmatter, body)`: the frontmatter as a dict
/// and the Markdown after it. Raises `ValueError` when there is no
/// frontmatter or its YAML is invalid.
#[pyfunction]
pub fn parse_skill_md<'py>(py: Python<'py>, text: &str) -> PyResult<(Bound<'py, PyAny>, String)> {
    let (frontmatter, body) = split_frontmatter(text).map_err(PyValueError::new_err)?;
    let fields = parse_frontmatter(frontmatter).map_err(PyValueError::new_err)?;
    Ok((json_to_python(py, &fields)?, body.to_string()))
}

/// Splits the text into the frontmatter (without its `---` lines) and the body.
fn split_frontmatter(text: &str) -> Result<(&str, &str), String> {
    // A byte order mark some editors add would hide the first `---`.
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let missing = || "SKILL.md must start with a frontmatter block between `---` lines".to_string();
    let first_line_end = text.find('\n').ok_or_else(missing)?;
    if text[..first_line_end].trim_end() != "---" {
        return Err(missing());
    }
    let after_opening = &text[first_line_end + 1..];
    // Look for the closing `---` line, keeping track of where each line starts.
    let mut line_start = 0;
    for line in after_opening.split_inclusive('\n') {
        if line.trim_end() == "---" {
            let frontmatter = &after_opening[..line_start];
            let body = &after_opening[line_start + line.len()..];
            return Ok((frontmatter, body.strip_prefix('\n').unwrap_or(body)));
        }
        line_start += line.len();
    }
    Err("The frontmatter of SKILL.md has no closing `---` line".to_string())
}

/// The frontmatter's YAML as a JSON object. When the YAML is invalid, tries
/// once more with `key: value: more` lines quoted (see `quote_values_with_colons`).
fn parse_frontmatter(frontmatter: &str) -> Result<Value, String> {
    match parse_yaml_mapping(frontmatter) {
        Ok(fields) => Ok(fields),
        Err(error) => parse_yaml_mapping(&quote_values_with_colons(frontmatter)).map_err(|_| error),
    }
}

fn parse_yaml_mapping(yaml: &str) -> Result<Value, String> {
    let documents =
        YamlLoader::load_from_str(yaml).map_err(|error| format!("Invalid YAML in the frontmatter: {error}"))?;
    match documents.into_iter().next() {
        // An empty frontmatter has no document at all.
        None | Some(Yaml::Null) => Ok(Value::Object(Map::new())),
        Some(document @ Yaml::Hash(_)) => Ok(yaml_to_json(document)),
        Some(_) => Err("The frontmatter must be a YAML mapping (`key: value` lines)".to_string()),
    }
}

/// Fixes the most common mistake in SKILL.md files, an unquoted value with a
/// colon in it, which YAML reads as a nested mapping:
///
/// ```text
/// description: Use when: the user asks for a PDF     (invalid YAML)
/// description: "Use when: the user asks for a PDF"   (what was meant)
/// ```
///
/// Only top-level lines whose value is plain text are touched.
fn quote_values_with_colons(frontmatter: &str) -> String {
    let mut fixed = String::with_capacity(frontmatter.len() + 16);
    for line in frontmatter.lines() {
        let top_level = !line.starts_with([' ', '\t', '#', '-']);
        match line.split_once(": ") {
            Some((key, value)) if top_level && is_plain_text_with_colon(value) => {
                let escaped = value.trim().replace('\\', "\\\\").replace('"', "\\\"");
                fixed.push_str(&format!("{key}: \"{escaped}\"\n"));
            }
            _ => {
                fixed.push_str(line);
                fixed.push('\n');
            }
        }
    }
    fixed
}

fn is_plain_text_with_colon(value: &str) -> bool {
    let value = value.trim();
    // Quoted strings, block scalars (`|`, `>`), lists, maps, anchors and tags are YAML already.
    let starts_like_yaml = value.starts_with(['"', '\'', '|', '>', '[', '{', '&', '*', '!']);
    !starts_like_yaml && value.contains(": ")
}

/// YAML -> JSON. Mapping keys become strings (YAML allows any key; JSON does not).
fn yaml_to_json(yaml: Yaml) -> Value {
    match yaml {
        Yaml::Null | Yaml::BadValue | Yaml::Alias(_) => Value::Null,
        Yaml::Boolean(boolean) => Value::Bool(boolean),
        Yaml::Integer(integer) => Value::from(integer),
        // A YAML float is kept as text until read; NaN and infinity have no JSON form.
        Yaml::Real(text) => text
            .parse::<f64>()
            .ok()
            .and_then(Number::from_f64)
            .map_or(Value::String(text), Value::Number),
        Yaml::String(text) => Value::String(text),
        Yaml::Array(items) => Value::Array(items.into_iter().map(yaml_to_json).collect()),
        Yaml::Hash(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key_text(key), yaml_to_json(value)))
                .collect(),
        ),
    }
}

/// A mapping key as text: `version: 1` and `1: x` both work.
fn key_text(key: Yaml) -> String {
    match key {
        Yaml::String(text) | Yaml::Real(text) => text,
        Yaml::Integer(integer) => integer.to_string(),
        Yaml::Boolean(boolean) => boolean.to_string(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(text: &str) -> Result<(Value, String), String> {
        let (frontmatter, body) = split_frontmatter(text)?;
        Ok((parse_frontmatter(frontmatter)?, body.to_string()))
    }

    #[test]
    fn reads_the_frontmatter_and_the_body() {
        let text = "---\nname: pdf\ndescription: Read PDFs.\nmetadata:\n  version: \"1.0\"\n---\n# PDF\nSteps.\n";
        let (fields, body) = parse(text).unwrap();
        assert_eq!(
            fields,
            json!({"name": "pdf", "description": "Read PDFs.", "metadata": {"version": "1.0"}})
        );
        assert_eq!(body, "# PDF\nSteps.\n");
    }

    #[test]
    fn accepts_windows_line_ends_and_block_scalars() {
        let text = "\u{feff}---\r\nname: pdf\r\ndescription: >\r\n  Read PDFs.\r\n  Use for forms.\r\n---\r\nBody";
        let (fields, body) = parse(text).unwrap();
        assert_eq!(fields["description"], json!("Read PDFs. Use for forms.\n"));
        assert_eq!(body, "Body");
    }

    #[test]
    fn quotes_unquoted_values_with_colons() {
        let text = "---\nname: pdf\ndescription: Use when: the user asks for a PDF\n---\n";
        let (fields, _) = parse(text).unwrap();
        assert_eq!(fields["description"], json!("Use when: the user asks for a PDF"));
    }

    #[test]
    fn rejects_files_without_frontmatter() {
        assert!(parse("# Just markdown\n").unwrap_err().contains("must start"));
        assert!(parse("---\nname: pdf\n").unwrap_err().contains("closing"));
        assert!(parse("---\n- a list\n---\n").unwrap_err().contains("mapping"));
        assert!(
            parse("---\nname: [unclosed\n---\n")
                .unwrap_err()
                .contains("Invalid YAML")
        );
    }

    #[test]
    fn an_empty_frontmatter_is_an_empty_mapping() {
        assert_eq!(parse("---\n---\nBody").unwrap(), (json!({}), "Body".to_string()));
    }
}
