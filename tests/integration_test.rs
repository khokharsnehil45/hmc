use hmc::storage::Storage;
use hmc::vector::{cosine_similarity, get_embedding};

#[test]
fn test_save_and_load() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let storage = Storage::with_path(db_path).unwrap();

    let content = "Conversation context: rust ownership and borrowing";
    let id = storage.save(content).unwrap();

    assert_eq!(id.len(), 8);

    let retrieved = storage.load(&id).unwrap();
    assert_eq!(retrieved, content);
}

#[test]
fn test_prefix_matching() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let storage = Storage::with_path(db_path).unwrap();

    let content = "Context to test prefix";
    let id = storage.save(content).unwrap();

    let retrieved = storage.load(&id[..4]).unwrap();
    assert_eq!(retrieved, content);
}

#[test]
fn test_vector_indexing_and_top_k_query() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let storage = Storage::with_path(db_path).unwrap();

    // 1. Rust doc
    let rust_doc = "Rust uses an ownership system with borrow checker for memory safety without garbage collection.";
    let id_rust = storage.save(rust_doc).unwrap();
    let rust_vec = get_embedding(rust_doc);
    storage.save_vector(&id_rust, &rust_vec).unwrap();

    // 2. Cooking doc
    let cooking_doc = "To make authentic carbonara, use eggs, pecorino romano cheese, guanciale and black pepper.";
    let id_cooking = storage.save(cooking_doc).unwrap();
    let cooking_vec = get_embedding(cooking_doc);
    storage.save_vector(&id_cooking, &cooking_vec).unwrap();

    // 3. PostgreSQL doc
    let pg_doc = "PostgreSQL provides JSONB with GIN index acceleration for relational document stores.";
    let id_pg = storage.save(pg_doc).unwrap();
    let pg_vec = get_embedding(pg_doc);
    storage.save_vector(&id_pg, &pg_vec).unwrap();

    // Query top 2 for "database json"
    let query_vec = get_embedding("database indexing with json");
    let results = storage.query_vectors(&query_vec, 2).unwrap();

    assert_eq!(results.len(), 2);
    // The top match should be PostgreSQL!
    assert_eq!(results[0].id, id_pg);
}

#[test]
fn test_cosine_similarity() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let c = vec![0.0, 1.0, 0.0];

    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);
    assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-5);
}
