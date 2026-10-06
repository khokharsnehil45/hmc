use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Deserialize)]
struct OllamaResponse {
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

pub fn get_embedding(text: &str) -> Vec<f32> {
    // Attempt Ollama first
    let ollama_url = std::env::var("HMC_OLLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:11434/api/embeddings".to_string());
    let ollama_model = std::env::var("HMC_OLLAMA_MODEL")
        .unwrap_or_else(|_| "nomic-embed-text".to_string());

    if let Ok(vec) = call_ollama(&ollama_url, &ollama_model, text) {
        if !vec.is_empty() {
            return vec;
        }
    }

    // Fallback to local deterministic feature-hashing embedding
    fallback_embedding(text)
}

fn call_ollama(url: &str, model: &str, prompt: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let req = OllamaRequest { model, prompt };
    let resp: OllamaResponse = ureq::post(url)
        .timeout(std::time::Duration::from_secs(4))
        .send_json(&req)?
        .into_json()?;
    Ok(resp.embedding)
}

/// 256-dimensional subword & word n-gram feature hashing embedding
fn fallback_embedding(text: &str) -> Vec<f32> {
    const DIM: usize = 256;
    let mut vec = vec![0.0f32; DIM];

    let words: Vec<String> = text
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect();

    for word in &words {
        // Hash whole word
        hash_feature(word, &mut vec, DIM);

        // Hash 3-character subword n-grams
        let chars: Vec<char> = word.chars().collect();
        if chars.len() >= 3 {
            for i in 0..=(chars.len() - 3) {
                let ngram: String = chars[i..i + 3].iter().collect();
                hash_feature(&ngram, &mut vec, DIM);
            }
        }
    }

    // L2 normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut vec {
            *x /= norm;
        }
    }
    vec
}

fn hash_feature(feat: &str, vec: &mut [f32], dim: usize) {
    let mut hasher = DefaultHasher::new();
    feat.hash(&mut hasher);
    let hash = hasher.finish();

    let index = (hash as usize) % dim;
    let sign = if (hash >> 32) % 2 == 0 { 1.0 } else { -1.0 };
    vec[index] += sign;
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
