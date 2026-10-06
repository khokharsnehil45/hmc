# hmc (Hold My Context)

A fast, lightweight CLI tool written in Rust to save and retrieve conversation contexts with SQLite-backed normalized storage and vector semantic search.

## Features

- **Relational Normalized Storage (SQLite):** Raw content is stored once in the `contexts` table. Vector embeddings are stored in a mapped `embeddings` table (`context_id -> vector BLOB`), preventing data duplication and accelerating similarity searches.
- **Save Context:** Save snippets via arguments or pipe from stdin (`hmc --save "..."`).
- **Vector Indexing:** Index contexts into vector store (`hmc --save "..." --vec`).
- **Semantic Querying:** Search contexts by meaning with top-k support (`hmc --query "..." --5` or `-k 5`).
- **Fast Local Embeddings:** Uses your local Ollama embeddings model (`nomic-embed-text`) if available, with automatic fallback to a built-in subword feature hashing vectorizer.
- **ID Retrieval:** Retrieve exact contexts by full ID or prefix (`hmc --load <id>`).

## Database Schema

Stored in `~/.hmc/hmc.db`:
```sql
CREATE TABLE contexts (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE embeddings (
    context_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    FOREIGN KEY(context_id) REFERENCES contexts(id) ON DELETE CASCADE
);
```

## Usage

### 1. Save Context
```bash
# Standard save (stored only in `contexts`)
hmc --save "Quick note on cargo test"

# Output:
# Saved context with ID: 10a3a569
```

### 2. Save with Vector Store Indexing
```bash
# Content stored in `contexts`, vector floats stored in `embeddings`
hmc --save "Understanding transformer self-attention mechanisms in LLMs" --vec

# Output:
# Saved context with ID: b2bd7dfa [vector indexed]
```

### 3. Query Top-K Matches
You can request the top *k* matches with shorthand flags like `--5`, `--10`, or `-k 5`:

```bash
# Get top 5 matches
hmc --query "artificial intelligence and machine learning" --5

# Get top 3 matches (default is 3)
hmc --query "baking and flour" --3

# Get top 1 match
hmc --query "baking and flour" --1
```

### 4. Retrieve Context by ID
```bash
hmc --load b2bd7dfa

# Or with prefix matching
hmc --load b2bd
```
