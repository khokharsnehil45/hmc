use clap::{CommandFactory, Parser};
use hmc::storage::Storage;
use hmc::vector::get_embedding;
use std::env;
use std::io::Read;
use std::process;

#[derive(Parser, Debug)]
#[command(
    name = "hmc",
    about = "Hold My Context (hmc) - Save and retrieve conversation contexts with vector store",
    version
)]
struct Cli {
    /// Save conversation context
    #[arg(short, long, value_name = "CONTENT")]
    save: Option<String>,

    /// Index context in vector store when saving
    #[arg(long)]
    vec: bool,

    /// Retrieve and print conversation context by ID
    #[arg(short, long, value_name = "ID")]
    load: Option<String>,

    /// Query vector store by semantic similarity
    #[arg(short, long, value_name = "QUERY")]
    query: Option<String>,

    /// Number of top matches to return (e.g. --5, -k 5, --top 5)
    #[arg(short = 'k', long = "top", default_value = "3")]
    top: usize,
}

fn preprocess_args<I: IntoIterator<Item = String>>(args: I) -> Vec<String> {
    let mut processed = Vec::new();
    for arg in args {
        // Support flags like --5, --10, --3 for top k
        if arg.starts_with("--") && arg.len() > 2 && arg[2..].chars().all(|c| c.is_ascii_digit()) {
            processed.push("--top".to_string());
            processed.push(arg[2..].to_string());
        } else {
            processed.push(arg);
        }
    }
    processed
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw_args: Vec<String> = env::args().collect();
    let processed_args = preprocess_args(raw_args);

    let cli = Cli::parse_from(processed_args);
    let storage = Storage::new()?;

    // 1. Save context
    if let Some(content) = cli.save {
        let text = if content == "-" {
            let mut buffer = String::new();
            std::io::stdin().read_to_string(&mut buffer)?;
            buffer
        } else {
            content
        };

        let id = storage.save(&text)?;

        if cli.vec {
            let embedding = get_embedding(&text);
            storage.save_vector(&id, &embedding)?;
            println!("Saved context with ID: {} [vector indexed]", id);
        } else {
            println!("Saved context with ID: {}", id);
        }
        return Ok(());
    }

    // 2. Load context by ID
    if let Some(id) = cli.load {
        match storage.load(&id) {
            Ok(content) => {
                print!("{}", content);
                if !content.ends_with('\n') {
                    println!();
                }
                return Ok(());
            }
            Err(err) => {
                eprintln!("Error: {}", err);
                process::exit(1);
            }
        }
    }

    // 3. Query vector store
    if let Some(query_text) = cli.query {
        let query_vec = get_embedding(&query_text);
        let results = storage.query_vectors(&query_vec, cli.top)?;

        if results.is_empty() {
            println!("No vector-indexed contexts found.");
            println!("Tip: Save contexts with --vec to enable vector search (e.g. hmc --save \"...\" --vec)");
            return Ok(());
        }

        println!("Top {} Vector Matches for: \"{}\"", results.len(), query_text);
        println!("{}", "=".repeat(65));

        for (idx, result) in results.iter().enumerate() {
            println!(
                "\n[Match #{}] ID: {} | Similarity: {:.2}",
                idx + 1,
                result.id,
                result.score
            );
            println!("{}", "-".repeat(65));
            println!("{}", result.content.trim_end());
        }
        return Ok(());
    }

    // No options provided: show help
    let mut cmd = Cli::command();
    cmd.print_help()?;
    println!();
    Ok(())
}
