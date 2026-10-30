//! Idle WebSocket connections, kept open between runs so the next run skips
//! the TCP, TLS and WebSocket handshakes (as reqwest's keep-alive pool does
//! for HTTP).
//!
//! A run takes a connection when it sends its first Responses API request,
//! keeps it for all its turns (the server's cache of recent responses lives on
//! that connection), and gives it back only after its last answer arrived
//! normally. A cancelled or failed run just drops its connection, which closes
//! it: that is also how the server learns to stop generating.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::client::ClientSettings;
use crate::ws::{OpenError, ResponsesSocket};

/// Servers close connections after 60 minutes; we stop using them a bit earlier,
/// between two turns, rather than have the server close one in the middle of a run.
const MAX_SOCKET_AGE: Duration = Duration::from_secs(55 * 60);
/// Idle connections older than this are closed instead of reused: the server's
/// idle timeout is not documented, and the next run can open a fresh one.
const MAX_IDLE_TIME: Duration = Duration::from_secs(60);
/// At most this many idle connections are kept; extra ones are closed.
const MAX_IDLE_SOCKETS: usize = 32;

/// Shared by every run of one `Client`.
///
/// A plain `Mutex` (a lock, like `threading.Lock` guarding the list) is
/// enough: it is held only to push or pop an entry, never while waiting on the network.
#[derive(Default)]
pub struct SocketPool {
    idle: Mutex<Vec<IdleSocket>>,
}

struct IdleSocket {
    socket: ResponsesSocket,
    idle_since: Instant,
}

impl SocketPool {
    /// An idle connection that is still young enough, or `None` (the caller opens a new one).
    fn take_idle(&self) -> Option<ResponsesSocket> {
        let now = Instant::now();
        let mut idle = self.idle.lock().unwrap();
        // `pop` takes the most recently used connection first: the least likely to be stale.
        while let Some(entry) = idle.pop() {
            if is_reusable(entry.socket.opened_at, entry.idle_since, now) {
                return Some(entry.socket);
            }
            // Otherwise `entry` is dropped here, which closes the connection.
        }
        None
    }

    /// Keeps a connection for the next run. Only call this when no response is in flight on it.
    fn give_back(&self, socket: ResponsesSocket) {
        let now = Instant::now();
        let mut idle = self.idle.lock().unwrap();
        if idle.len() < MAX_IDLE_SOCKETS && is_reusable(socket.opened_at, now, now) {
            idle.push(IdleSocket {
                socket,
                idle_since: now,
            });
        }
    }
}

/// A pure function, so the rules can be tested without waiting an hour.
fn is_reusable(opened_at: Instant, idle_since: Instant, now: Instant) -> bool {
    now - opened_at < MAX_SOCKET_AGE && now - idle_since < MAX_IDLE_TIME
}

/// The connection one run uses for all its turns. Empty until the run's first
/// Responses API request over WebSocket.
///
/// `Option` is Rust's "maybe a value": `Some(socket)` or `None`. When the run
/// ends, Rust drops (frees) this value, which closes the connection, unless
/// the run gave it back to the pool first with `give_back`.
#[derive(Default)]
pub struct RunSocket {
    socket: Option<ResponsesSocket>,
}

impl RunSocket {
    /// The run's connection, ready for the next turn: the one it already
    /// holds, an idle one from the pool, or a new one.
    pub async fn connect(&mut self, client: &ClientSettings) -> Result<&mut ResponsesSocket, OpenError> {
        if self
            .socket
            .as_ref()
            .is_some_and(|socket| socket.opened_at.elapsed() >= MAX_SOCKET_AGE)
        {
            self.close(); // too old: switch connections between two turns
        }
        if self.socket.is_none() {
            let socket = match client.websocket_pool.take_idle() {
                Some(socket) => socket,
                None => ResponsesSocket::open(client).await?, // `?`: on error, return it to our caller
            };
            self.socket = Some(socket);
        }
        Ok(self.socket.as_mut().expect("a socket was just set"))
    }

    /// Closes the run's connection, after a failure or with a response still in flight.
    pub fn close(&mut self) {
        self.socket = None; // dropping the old value closes it
    }

    /// Gives the connection to the pool for the next run. Call it only when
    /// the run's last response finished normally.
    pub fn give_back(self, client: &ClientSettings) {
        if let Some(mut socket) = self.socket {
            // The next run sends its whole input on its first turn: the server
            // may have dropped this response from its cache while the
            // connection sat idle (Azure keeps only one), which would cost a
            // failed request.
            socket.cached_response_id = None;
            client.websocket_pool.give_back(socket);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_connections_are_reused_only_while_young_and_recently_used() {
        let opened_at = Instant::now();
        let minutes = |count: u64| Duration::from_secs(count * 60);

        // Used half a minute ago, opened 10 minutes ago.
        assert!(is_reusable(
            opened_at,
            opened_at + minutes(10),
            opened_at + minutes(10) + minutes(1) / 2
        ));
        // Idle for more than a minute.
        assert!(!is_reusable(opened_at, opened_at, opened_at + minutes(2)));
        // Used a second ago, but too close to the server's 60-minute limit.
        let now = opened_at + minutes(56);
        assert!(!is_reusable(opened_at, now - Duration::from_secs(1), now));
    }
}
