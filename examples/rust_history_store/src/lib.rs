//! Example: a conversation history store written in Rust, keeping
//! conversations in SQLite through sqlx.
//!
//! The agent calls `load` before each run and `save` after it, directly in
//! Rust: no Python runs and the messages never become Python objects. The
//! connection pool is created once, in `sqlite_history`, and shared by every
//! run of every agent using this store.
//!
//! For Postgres, use `PgPool` / `PgPoolOptions`, `$1`-style placeholders and
//! a `JSONB` column; the rest stays the same.

use std::str::FromStr;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyox_tools::history::{HistoryStore, RustHistoryStore, runtime};
use pyox_tools::messages::Message;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions};

/// One row per message, so saving a run only appends its new messages.
/// The primary key also serves as the index `load` uses.
const CREATE_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS messages (
        conversation_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        message TEXT NOT NULL, -- the message as JSON
        PRIMARY KEY (conversation_id, position)
    )";

struct SqliteHistory {
    pool: SqlitePool,
}

impl HistoryStore for SqliteHistory {
    async fn load(&self, conversation_id: &str) -> Result<Vec<Message>, String> {
        let rows: Vec<String> =
            sqlx::query_scalar("SELECT message FROM messages WHERE conversation_id = ? ORDER BY position")
                .bind(conversation_id)
                .fetch_all(&self.pool)
                .await
                .map_err(|error| error.to_string())?;
        rows.iter()
            .map(|row| serde_json::from_str(row).map_err(|error| error.to_string()))
            .collect()
    }

    async fn save(&self, conversation_id: &str, new_messages: &[Message]) -> Result<(), String> {
        // A transaction: either all the new messages are saved, or none.
        let mut transaction = self.pool.begin().await.map_err(|error| error.to_string())?;
        for message in new_messages {
            let message_json = serde_json::to_string(message).expect("messages always serialize");
            // The position is computed in the same statement: starting with a
            // write makes SQLite lock the database at once, so concurrent saves
            // wait their turn instead of failing.
            sqlx::query(
                "INSERT INTO messages (conversation_id, position, message)
                 SELECT ?1, COALESCE(MAX(position) + 1, 0), ?2 FROM messages WHERE conversation_id = ?1",
            )
            .bind(conversation_id)
            .bind(message_json)
            .execute(&mut *transaction)
            .await
            .map_err(|error| error.to_string())?;
        }
        transaction.commit().await.map_err(|error| error.to_string())
    }
}

/// Python: `fast_history.sqlite_history("sqlite://chats.db", max_connections=8)`.
///
/// Opens (or creates) the database and its connection pool. Call it once at
/// startup and give the store to your agents.
#[pyfunction]
#[pyo3(signature = (url, *, max_connections = 8))]
fn sqlite_history(py: Python<'_>, url: &str, max_connections: u32) -> PyResult<RustHistoryStore> {
    let connect = async {
        let options = SqliteConnectOptions::from_str(url)?
            .create_if_missing(true)
            // WAL lets runs read while another run writes.
            .journal_mode(SqliteJournalMode::Wal);
        let pool = SqlitePoolOptions::new()
            .max_connections(max_connections)
            .connect_with(options)
            .await?;
        sqlx::query(CREATE_TABLE).execute(&pool).await?;
        Ok::<_, sqlx::Error>(pool)
    };
    // The pool must be created inside the store's runtime, which also runs
    // its background tasks. `detach` releases Python's global lock meanwhile.
    let pool = py
        .detach(|| runtime().block_on(connect))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    RustHistoryStore::new(SqliteHistory { pool })
}

/// The Python module: `import fast_history`.
#[pymodule]
fn fast_history(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sqlite_history, module)?)
}
