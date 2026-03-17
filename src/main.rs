mod manifold;
use manifold::{LeibnizManifold};
use std::io::{self, Write};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = sled::open("inference_vault")?;
    let mut manifold = load_manifold(&db)?;

    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("ingest") => {
            let path = args.get(2).expect("Provide file path");
            ingest_file(path, &mut manifold, &db)?;
        }
        Some("chat") => {
            println!("DAEMON V5 | 64K DIM | DOT-PRODUCT QUERY ACTIVE");
            loop {
                print!("User >> ");
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let trimmed = input.trim();
                if trimmed.is_empty() { break; }
                process_query(&manifold, trimmed);
            }
        }
        Some("diagnostics") | Some("diag") => {
            println!("--- MANIFOLD DIAGNOSTICS ---");
            println!("Saturation: {:.2}%", manifold.get_saturation() * 100.0);
            println!("Vocabulary: {} tokens", manifold.dictionary.len());
        }
        _ => println!("Usage: leibniz_node [ingest|chat|diagnostics]"),
    }

    save_manifold(&db, &manifold)?;
    Ok(())
}

fn process_query(manifold: &LeibnizManifold, query: &str) {
    let results = manifold.query(query);
    if results.is_empty() {
        println!("Response: [No signal]");
    } else {
        print!("Response: ");
        for (word, score) in results {
            print!("{} ({:.1}) ", word, score);
        }
        println!();
    }
}

fn ingest_file(path: &str, manifold: &mut LeibnizManifold, db: &sled::Db) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let file_hash = blake3::hash(content.as_bytes()).to_string();
    
    // Check-sum feature: skip duplicates
    if db.get(format!("hash_{}", file_hash))?.is_some() {
        println!("Check-sum Match: Skipping {}", path);
        return Ok(());
    }

    println!("Ingesting: {}", path);
    for word in content.split_whitespace() {
        let clean = word.to_lowercase().replace(|c: char| !c.is_alphanumeric(), "");
        if !clean.is_empty() {
            manifold.ingest_token(&clean);
        }
    }

    db.insert(format!("hash_{}", file_hash), &[1])?;
    Ok(())
}

fn load_manifold(db: &sled::Db) -> Result<LeibnizManifold, Box<dyn std::error::Error>> {
    if let Some(bytes) = db.get("manifold_state")? {
        Ok(bincode::deserialize(&bytes)?)
    } else {
        Ok(LeibnizManifold::new())
    }
}

fn save_manifold(db: &sled::Db, manifold: &LeibnizManifold) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = bincode::serialize(manifold)?;
    db.insert("manifold_state", encoded)?;
    db.flush()?;
    Ok(())
}