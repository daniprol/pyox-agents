//! Server-sent events (SSE) decoding.
//!
//! A streaming response is a sequence of lines like:
//!
//! ```text
//! data: {"choices":[...]}
//!
//! data: [DONE]
//! ```
//!
//! Network chunks can end in the middle of a line (or even in the middle of
//! a multi-byte UTF-8 character), so we keep unfinished bytes in a buffer
//! until the rest arrives.

/// Collects bytes and returns the `data:` payloads of complete events.
#[derive(Default)]
pub struct SseDecoder {
    /// Bytes received but not yet processed (an unfinished line).
    pending_bytes: Vec<u8>,
    /// `data:` lines of the event being read. An event can span several lines.
    current_event_data: Vec<String>,
}

impl SseDecoder {
    /// Feeds a network chunk; returns the data of every event it completed.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.pending_bytes.extend_from_slice(chunk);
        let mut finished_events = Vec::new();

        while let Some(newline_position) = self.pending_bytes.iter().position(|&byte| byte == b'\n') {
            let line_bytes: Vec<u8> = self.pending_bytes.drain(..=newline_position).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches(['\n', '\r']);

            if line.is_empty() {
                // A blank line ends the current event.
                if !self.current_event_data.is_empty() {
                    finished_events.push(self.current_event_data.join("\n"));
                    self.current_event_data.clear();
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                self.current_event_data
                    .push(data.strip_prefix(' ').unwrap_or(data).to_string());
            }
            // Other fields (`event:`, `id:`, `retry:`) and comments (`:`) are not used by the API.
        }

        finished_events
    }

    /// Call when the stream ends, to get an event not followed by a blank line.
    pub fn finish(&mut self) -> Option<String> {
        let leftover = self.push(b"\n\n");
        leftover.into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_split_across_chunks_are_joined() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"data: {\"a\"").is_empty());
        assert_eq!(decoder.push(b":1}\n\ndata: [DONE]\n\n"), vec!["{\"a\":1}", "[DONE]"]);
    }

    #[test]
    fn utf8_characters_split_across_chunks_survive() {
        let mut decoder = SseDecoder::default();
        let bytes = "data: ñ\n\n".as_bytes();
        assert!(decoder.push(&bytes[..7]).is_empty()); // cuts "ñ" in half
        assert_eq!(decoder.push(&bytes[7..]), vec!["ñ"]);
    }

    #[test]
    fn crlf_comments_and_unterminated_events() {
        let mut decoder = SseDecoder::default();
        assert_eq!(decoder.push(b": keep-alive\r\n\r\ndata: x\r\n\r\n"), vec!["x"]);
        assert!(decoder.push(b"data: last").is_empty());
        assert_eq!(decoder.finish(), Some("last".to_string()));
    }
}
