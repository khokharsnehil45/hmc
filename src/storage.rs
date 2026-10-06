use crate::vector::cosine_similarity;
use chrono::Local;
use rusqlite::{params, Connection};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub id: String,
    pub score: f32,
    pub content: String,
}

pub struct Storage {
    conn: Connection,
    #[allow(dead_code)]
    db_path: PathBuf,
}

impl Storage {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let base_dir = if let Ok(custom) = std::env::var("HMC_DIR") {
            PathBuf::from(custom)
        } else if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".hmc")
        } else {
            PathBuf::from(".hmc")
        };

        fs::create_dir_all(&base_dir)?;
        let db_path = base_dir.join("hmc.db");
        Self::with_path(db_path)
    }

    pub fn with_path(db_path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path)?;

        // Enable foreign keys
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS contexts (
                 id TEXT PRIMARY KEY,
                 content TEXT NOT NULL,
                 created_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS embeddings (
                 context_id TEXT PRIMARY KEY,
                 vector BLOB NOT NULL,
                 FOREIGN KEY(context_id) REFERENCES contexts(id) ON DELETE CASCADE
             );"
        )?;

        let storage = Self { conn, db_path };
        storage.migrate_legacy_files();

        Ok(storage)
    }

    fn migrate_legacy_files(&self) {
        if let Some(parent) = self.db_path.parent() {
            if let Ok(entries) = fs::read_dir(parent) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
                        if let Some(id) = path.file_stem().and_then(|s| s.to_str()) {
                            if let Ok(content) = fs::read_to_string(&path) {
                                let _ = self.save_with_id(id, &content);

                                // Check if matching .vec file exists
                                let vec_path = parent.join(format!("{}.vec", id));
                                if vec_path.exists() {
                                    if let Ok(vec_str) = fs::read_to_string(&vec_path) {
                                        if let Ok(vec) = serde_json::from_str::<Vec<f32>>(&vec_str) {
                                            let _ = self.save_vector(id, &vec);
                                        }
                                    }
                                    let _ = fs::remove_file(vec_path);
                                }
                            }
                            let _ = fs::remove_file(path);
                        }
                    }
                }
            }
        }
    }

    fn save_with_id(&self, id: &str, content: &str) -> Result<(), rusqlite::Error> {
        let now = Local::now().to_rfc3339();
        self.conn.execute(
            "INSERT OR REPLACE INTO contexts (id, content, created_at) VALUES (?1, ?2, ?3)",
            params![id, content, now],
        )?;
        Ok(())
    }

    pub fn save(&self, content: &str) -> Result<String, Box<dyn std::error::Error>> {
        let id = self.generate_unique_id();
        self.save_with_id(&id, content)?;
        Ok(id)
    }

    pub fn save_vector(&self, id: &str, vector: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        let blob = encode_vector(vector);
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (context_id, vector) VALUES (?1, ?2)",
            params![id, blob],
        )?;
        Ok(())
    }

    pub fn load(&self, id_prefix: &str) -> Result<String, String> {
        // 1. Exact match
        let mut stmt = self
            .conn
            .prepare("SELECT content FROM contexts WHERE id = ?1")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt
            .query_map(params![id_prefix], |row| row.get::<_, String>(0))
            .map_err(|e| e.to_string())?;

        if let Some(row) = rows.next() {
            return row.map_err(|e| e.to_string());
        }

        // 2. Prefix match
        let prefix_lower = id_prefix.to_lowercase();
        let mut stmt = self
            .conn
            .prepare("SELECT id, content FROM contexts WHERE LOWER(id) LIKE ?1 || '%'")
            .map_err(|e| e.to_string())?;

        let matches: Vec<(String, String)> = stmt
            .query_map(params![prefix_lower], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .collect();

        if matches.is_empty() {
            return Err(format!("No context found with ID matching '{}'", id_prefix));
        }

        if matches.len() > 1 {
            let matched_ids: Vec<String> = matches.into_iter().map(|(id, _)| id).collect();
            return Err(format!(
                "Ambiguous ID prefix '{}'. Multiple contexts matched: {}",
                id_prefix,
                matched_ids.join(", ")
            ));
        }

        Ok(matches.into_iter().next().unwrap().1)
    }

    pub fn query_vectors(
        &self,
        query_vec: &[f32],
        top_k: usize,
    ) -> Result<Vec<QueryResult>, Box<dyn std::error::Error>> {
        // STEP 1: Scan ONLY the embeddings table (memory-efficient: no text loaded)
        let mut stmt = self
            .conn
            .prepare("SELECT context_id, vector FROM embeddings")?;

        let mut scored_ids: Vec<(String, f32)> = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob))
            })?
            .filter_map(Result::ok)
            .map(|(id, blob)| {
                let doc_vec = decode_vector(&blob);
                let score = cosine_similarity(query_vec, &doc_vec);
                (id, score)
            })
            .collect();

        // Sort descending by score
        scored_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k
        let top_winners: Vec<(String, f32)> = scored_ids.into_iter().take(top_k).collect();

        // STEP 2: Fetch actual text ONLY for the winning IDs from contexts table
        let mut results = Vec::new();
        for (id, score) in top_winners {
            let content: Option<String> = self
                .conn
                .query_row(
                    "SELECT content FROM contexts WHERE id = ?1",
                    params![id],
                    |row| row.get(0),
                )
                .ok();

            if let Some(content) = content {
                results.push(QueryResult { id, score, content });
            }
        }

        Ok(results)
    }

    fn generate_unique_id(&self) -> String {
        loop {
            let id = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
            let exists: bool = self
                .conn
                .query_row(
                    "SELECT 1 FROM contexts WHERE id = ?1",
                    params![id],
                    |_| Ok(true),
                )
                .unwrap_or(false);

            if !exists {
                return id;
            }
        }
    }
}

fn encode_vector(vec: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vec.len() * 4);
    for &val in vec {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn decode_vector(bytes: &[u8]) -> Vec<f32> {
    let mut vec = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        vec.push(val);
    }
    vec
}
