//! How streamed text is grouped before it is sent to Python.
//!
//! Models stream text in tiny pieces (often a word or less). Handing each
//! piece to Python as a separate `TextDelta` costs CPU, so users can choose
//! bigger chunks:
//!
//! | Mode            | One `TextDelta` per...                                      |
//! |-----------------|-------------------------------------------------------------|
//! | `Delta`         | piece received from the API (default, lowest latency)       |
//! | `EveryNDeltas`  | N pieces                                                    |
//! | `Paragraph`     | markdown paragraph; a fenced code block is never split       |
//! | `Message`       | whole assistant message (all text between tool calls)       |
//!
//! This sits between the API adapter and the event queue, so neither the
//! adapter nor the agent loop needs to know which mode is used.
//!
//! The paragraph mode looks only at complete lines (like Codex's terminal UI,
//! which only commits text up to the last newline): a paragraph ends at a
//! blank line, except inside a ``` or ~~~ code block, which is sent as a whole
//! once its closing fence arrives.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tokio::sync::mpsc::Sender;

use crate::events::{AgentEvent, TextDelta};

/// The grouping mode, chosen in Python with `text_chunks="delta"`,
/// `"paragraph"`, `"message"` or a number N.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextChunking {
    Delta,
    EveryNDeltas(usize),
    Paragraph,
    Message,
}

impl TextChunking {
    pub fn from_python(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(count) = value.extract::<i64>() {
            return match count {
                ..=0 => Err(PyValueError::new_err("text_chunks must be at least 1")),
                1 => Ok(TextChunking::Delta),
                _ => Ok(TextChunking::EveryNDeltas(count as usize)),
            };
        }
        match value.extract::<String>()?.as_str() {
            "delta" => Ok(TextChunking::Delta),
            "paragraph" => Ok(TextChunking::Paragraph),
            "message" => Ok(TextChunking::Message),
            other => Err(PyValueError::new_err(format!(
                "text_chunks must be \"delta\", \"paragraph\", \"message\" or a number, not {other:?}"
            ))),
        }
    }
}

/// Collects text pieces and says when a chunk is ready to send.
pub struct TextChunker {
    mode: TextChunking,
    /// Text received but not sent yet.
    pending: String,
    /// How many pieces are in `pending` (for `EveryNDeltas`).
    pending_pieces: usize,
    /// Paragraph mode: where the first line not yet looked at starts in `pending`.
    unchecked_from: usize,
    /// Paragraph mode: the fence (e.g. "```") of the code block we are inside, if any.
    open_code_fence: Option<String>,
}

impl TextChunker {
    pub fn new(mode: TextChunking) -> Self {
        TextChunker {
            mode,
            pending: String::new(),
            pending_pieces: 0,
            unchecked_from: 0,
            open_code_fence: None,
        }
    }

    /// Adds a piece of text. Returns a chunk if one is ready.
    pub fn push(&mut self, piece: &str) -> Option<String> {
        if piece.is_empty() {
            return None;
        }
        self.pending.push_str(piece);
        self.pending_pieces += 1;

        match self.mode {
            TextChunking::Delta => self.take_pending(),
            TextChunking::EveryNDeltas(count) if self.pending_pieces >= count => self.take_pending(),
            TextChunking::EveryNDeltas(_) | TextChunking::Message => None,
            TextChunking::Paragraph => self.take_finished_paragraphs(),
        }
    }

    /// The message ended (or a tool call is next): returns whatever is left.
    pub fn finish(&mut self) -> Option<String> {
        self.open_code_fence = None;
        self.take_pending()
    }

    fn take_pending(&mut self) -> Option<String> {
        self.pending_pieces = 0;
        self.unchecked_from = 0;
        if self.pending.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.pending))
        }
    }

    /// Looks at each new complete line; returns the text up to the last line
    /// that ends a paragraph or a code block.
    fn take_finished_paragraphs(&mut self) -> Option<String> {
        let mut chunk_end = None;

        while let Some(newline_offset) = self.pending[self.unchecked_from..].find('\n') {
            let line_end = self.unchecked_from + newline_offset + 1;
            let line = &self.pending[self.unchecked_from..line_end];
            self.unchecked_from = line_end;

            let line_ends_a_block = match (&self.open_code_fence, code_fence_of(line)) {
                // A fence line closes the code block if it uses the same characters, at least as many.
                (Some(open_fence), Some(fence)) if fence.starts_with(open_fence.as_str()) => {
                    self.open_code_fence = None;
                    true
                }
                (Some(_), _) => false,
                (None, Some(fence)) => {
                    self.open_code_fence = Some(fence);
                    false
                }
                (None, None) => line.trim().is_empty(),
            };
            if line_ends_a_block {
                chunk_end = Some(line_end);
            }
        }

        let chunk_end = chunk_end?;
        if self.pending[..chunk_end].trim().is_empty() {
            return None; // only blank lines so far: keep them with the next paragraph
        }
        let rest = self.pending.split_off(chunk_end);
        let chunk = std::mem::replace(&mut self.pending, rest);
        self.unchecked_from -= chunk_end;
        self.pending_pieces = 0;
        Some(chunk)
    }
}

/// Where API adapters write streamed text: groups it with a `TextChunker`
/// and sends each chunk to Python as a `TextDelta` event.
pub struct TextOutput {
    chunker: TextChunker,
    event_sender: Sender<PyResult<AgentEvent>>,
}

impl TextOutput {
    pub fn new(mode: TextChunking, event_sender: Sender<PyResult<AgentEvent>>) -> Self {
        TextOutput {
            chunker: TextChunker::new(mode),
            event_sender,
        }
    }

    /// Adds a piece of streamed text; sends a chunk if one is ready.
    pub async fn push(&mut self, piece: &str) {
        if let Some(text) = self.chunker.push(piece) {
            self.send(text).await;
        }
    }

    /// Sends what is left. Called when a model message ends.
    pub async fn finish(&mut self) {
        if let Some(text) = self.chunker.finish() {
            self.send(text).await;
        }
    }

    async fn send(&self, text: String) {
        // If Python stopped listening, sending fails; the run is being cancelled anyway.
        let _ = self
            .event_sender
            .send(Ok(AgentEvent::TextDelta(TextDelta { text })))
            .await;
    }
}

/// If `line` starts or ends a fenced code block, returns its fence ("```", "~~~~", ...).
fn code_fence_of(line: &str) -> Option<String> {
    let indentation = line.len() - line.trim_start_matches(' ').len();
    if indentation > 3 {
        return None; // markdown treats 4+ spaces as an indented code line, not a fence
    }
    let text = line.trim();
    let fence_character = text
        .chars()
        .next()
        .filter(|character| *character == '`' || *character == '~')?;
    let fence_length = text
        .chars()
        .take_while(|character| *character == fence_character)
        .count();
    (fence_length >= 3).then(|| fence_character.to_string().repeat(fence_length))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds pieces one by one and returns every chunk produced, including the final one.
    fn chunks(mode: TextChunking, pieces: &[&str]) -> Vec<String> {
        let mut chunker = TextChunker::new(mode);
        let mut produced: Vec<String> = pieces.iter().filter_map(|piece| chunker.push(piece)).collect();
        produced.extend(chunker.finish());
        produced
    }

    #[test]
    fn delta_mode_passes_pieces_through() {
        assert_eq!(chunks(TextChunking::Delta, &["Hel", "lo", ""]), ["Hel", "lo"]);
    }

    #[test]
    fn every_n_deltas_groups_pieces() {
        assert_eq!(
            chunks(TextChunking::EveryNDeltas(2), &["a", "b", "c", "d", "e"]),
            ["ab", "cd", "e"]
        );
    }

    #[test]
    fn message_mode_sends_everything_at_the_end() {
        assert_eq!(chunks(TextChunking::Message, &["a", "b\n\n", "c"]), ["ab\n\nc"]);
    }

    #[test]
    fn paragraphs_end_at_blank_lines() {
        let pieces = ["First para", "graph.\nSame one.\n", "\nSecond", " one.\n\n\n", "Third"];
        assert_eq!(
            chunks(TextChunking::Paragraph, &pieces),
            ["First paragraph.\nSame one.\n\n", "Second one.\n\n\n", "Third"]
        );
    }

    #[test]
    fn code_blocks_are_never_split() {
        let pieces = ["Code:\n\n```py", "thon\na = 1\n\nb = 2\n", "```\n", "Done."];
        assert_eq!(
            chunks(TextChunking::Paragraph, &pieces),
            ["Code:\n\n", "```python\na = 1\n\nb = 2\n```\n", "Done."]
        );
    }

    #[test]
    fn a_fence_only_closes_with_the_same_kind() {
        let text = "~~~~\n```\n\n~~~\n\n~~~~\nafter\n\n";
        assert_eq!(
            chunks(TextChunking::Paragraph, &[text]),
            ["~~~~\n```\n\n~~~\n\n~~~~\nafter\n\n"]
        );
    }

    #[test]
    fn leading_blank_lines_stay_with_the_next_paragraph() {
        assert_eq!(chunks(TextChunking::Paragraph, &["\n\n", "Hi\n\n"]), ["\n\nHi\n\n"]);
    }
}
