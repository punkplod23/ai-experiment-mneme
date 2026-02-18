use bitvec::prelude::*;
use clap::{Parser, Subcommand};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use tiny_keccak::{Hasher, Sha3};

const DIMENSIONS: usize = 10000;
type HyperVector = BitVec<u64, Lsb0>;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Dictionary { path: String },
    Ingest { 
        path: String, 
        #[arg(short, long, default_value_t = 0)] label_id: u32 
    },
    Chat { 
        seed_phrase: String, 
        #[arg(short, long, default_value_t = 100)] length: usize, 
        #[arg(short, long, default_value_t = 0.5)] temp: f32, 
        #[arg(short, long, default_value_t = 0)] label_id: u32 
    },
    Prune { #[arg(short, long, default_value_t = 1)] threshold: u32 },
}

// --- HDC Core & Variable Complexity Logic ---

fn get_complexity_multiplier(label_id: u32) -> usize {
    match label_id {
        2 | 5 | 7 => 2, // Double resolution for Philosophy, Modern Science, and Computation
        _ => 1,
    }
}

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn generate_id_vec(text: &str) -> HyperVector {
    let mut bits = BitVec::<u64, Lsb0>::repeat(false, DIMENSIONS);
    let clean = text.trim();
    if clean.is_empty() { return bits; }
    let mut hasher = Sha3::v256();
    hasher.update(clean.as_bytes());
    let mut output = [0u8; 32];
    hasher.finalize(&mut output);
    for (i, byte) in output.iter().enumerate() {
        for bit in 0..8 {
            if (byte >> bit) & 1 == 1 {
                bits.set((i * 8 + bit) % DIMENSIONS, true);
            }
        }
    }
    bits
}

// NEW: Permutation-based expansion for Variable Dimensionality
fn generate_complex_id_vec(text: &str, multiplier: usize) -> HyperVector {
    let mut base = generate_id_vec(text);
    if multiplier > 1 {
        for i in 1..multiplier {
            let mut shifted = base.clone();
            shifted.rotate_left((i * 137) % DIMENSIONS); // Prime-step rotation
            base ^= shifted; // Bind high-order features via XOR
        }
    }
    base
}

fn shift_left(bv: &HyperVector, amount: usize) -> HyperVector {
    let mut result = bv.clone();
    result.rotate_left(amount % DIMENSIONS);
    result
}

fn bundle(vectors: &[HyperVector]) -> HyperVector {
    if vectors.is_empty() { return BitVec::<u64, Lsb0>::repeat(false, DIMENSIONS); }
    let mut counts = vec![0i32; DIMENSIONS];
    for v in vectors {
        for i in 0..DIMENSIONS {
            if v[i] { counts[i] += 1; } else { counts[i] -= 1; }
        }
    }
    let mut result = BitVec::<u64, Lsb0>::repeat(false, DIMENSIONS);
    for i in 0..DIMENSIONS {
        if counts[i] > 0 { result.set(i, true); }
    }
    result
}

fn bv_to_bytes(bv: &HyperVector) -> Vec<u8> {
    bv.as_raw_slice().iter().flat_map(|&u| u.to_ne_bytes()).collect()
}

fn bytes_to_bv(bytes: &[u8]) -> HyperVector {
    let mut raw = Vec::new();
    for chunk in bytes.chunks_exact(8) {
        raw.push(u64::from_ne_bytes(chunk.try_into().unwrap()));
    }
    let mut bv = BitVec::<u64, Lsb0>::from_vec(raw);
    bv.truncate(DIMENSIONS);
    bv
}

fn get_id(vocab: &sled::Tree, word: &str) -> u32 {
    vocab.get(word.as_bytes()).ok().flatten()
        .map(|v| u32::from_be_bytes(v.as_ref().try_into().unwrap_or([0; 4])))
        .unwrap_or(0)
}

fn get_or_create_id(vocab: &sled::Tree, word: &str, db: &sled::Db) -> u32 {
    let clean = normalize(word);
    if clean.is_empty() || clean.contains("gutenberg") { return 0; }
    if let Some(val) = vocab.get(clean.as_bytes()).unwrap() {
        u32::from_be_bytes(val.as_ref().try_into().unwrap_or([0; 4]))
    } else {
        let id = db.generate_id().unwrap() as u32;
        let _ = vocab.insert(clean.as_bytes(), &id.to_be_bytes());
        let _ = vocab.insert(&id.to_be_bytes(), clean.as_bytes());
        id
    }
}

// --- Main Execution ---

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    
    let db = sled::Config::new()
        .path("sovereign_vault")
        .cache_capacity(1_073_741_824) 
        .use_compression(true)
        .open()
        .expect("Failed to open vault");

    let links = db.open_tree("links").unwrap();
    let vocab = db.open_tree("vocab").unwrap();
    let vectors = db.open_tree("vectors").unwrap();
    let centroids = db.open_tree("centroids").unwrap();
    let weights = db.open_tree("weights").unwrap();

    match cli.command {
        Commands::Dictionary { path } => {
            let reader = BufReader::new(File::open(&path)?);
            for line in reader.lines().map_while(Result::ok) {
                for word in line.split_whitespace() {
                    let id = get_or_create_id(&vocab, word, &db);
                    if id != 0 {
                        // Default dictionary anchoring at multiplier 1
                        let v = generate_id_vec(word);
                        let _ = vectors.insert(&id.to_be_bytes(), bv_to_bytes(&v));
                    }
                }
            }
        }

        Commands::Ingest { path, label_id } => {
            let reader = BufReader::new(File::open(&path)?);
            let mut sample_vecs = Vec::new();
            let multiplier = get_complexity_multiplier(label_id);

            for line in reader.lines().map_while(Result::ok) {
                let tokens: Vec<u32> = line.split_whitespace()
                    .map(|s| get_or_create_id(&vocab, s, &db))
                    .filter(|id| *id != 0).collect();

                for window in tokens.windows(4) {
                    let key = [
                        label_id.to_be_bytes(), 
                        window[0].to_be_bytes(), 
                        window[1].to_be_bytes(),
                        window[2].to_be_bytes()
                    ].concat();
                    
                    let _ = links.insert(&key, &window[3].to_be_bytes());
                    let cur_w = weights.get(&key).unwrap().map(|v| u32::from_be_bytes(v.as_ref().try_into().unwrap_or([0;4]))).unwrap_or(0);
                    let _ = weights.insert(&key, &(cur_w + 1).to_be_bytes());

                    if let Some(wb) = vocab.get(&window[3].to_be_bytes()).unwrap() {
                        let word = String::from_utf8_lossy(&wb);
                        // Store the vector using the complexity multiplier for this silo
                        let v = generate_complex_id_vec(&word, multiplier);
                        let _ = vectors.insert(&window[3].to_be_bytes(), bv_to_bytes(&v));
                        if sample_vecs.len() < 2000 { sample_vecs.push(v); }
                    }
                }
            }
            let _ = centroids.insert(&label_id.to_be_bytes(), bv_to_bytes(&bundle(&sample_vecs)));
            println!("Ingested (Complexity x{}): {} into Label {}", multiplier, path, label_id);
        }

        Commands::Chat { seed_phrase, length, temp, label_id } => {
            let mut history = Vec::new();
            let seeds: Vec<u32> = seed_phrase.split_whitespace()
                .map(|s| get_id(&vocab, &normalize(s)))
                .filter(|id| *id != 0).collect();

            if seeds.is_empty() { return Ok(()); }

            let multiplier = get_complexity_multiplier(label_id);
            let centroid = if label_id != 0 {
                centroids.get(&label_id.to_be_bytes()).unwrap()
                    .map(|v| bytes_to_bv(&v))
                    .unwrap_or_else(|| BitVec::<u64, Lsb0>::repeat(false, DIMENSIONS))
            } else { BitVec::<u64, Lsb0>::repeat(false, DIMENSIONS) };

            for &id in &seeds {
                history.push(id);
                if let Some(wb) = vocab.get(&id.to_be_bytes()).unwrap() {
                    print!("{} ", String::from_utf8_lossy(&wb));
                }
            }
            io::stdout().flush()?;

            for i in 0..length {
                let mut candidates = Vec::new();

                if history.len() >= 3 {
                    let w1 = history[history.len()-3];
                    let w2 = history[history.len()-2];
                    let w3 = history[history.len()-1];
                    let target_hist = [w1.to_be_bytes(), w2.to_be_bytes(), w3.to_be_bytes()].concat();
                    
                    if label_id == 0 {
                        let all_links: Vec<_> = links.iter().filter_map(Result::ok).collect();
                        let found_matches: Vec<u32> = all_links.into_par_iter()
                            .filter(|(k, _)| k.len() == 16 && &k[4..16] == target_hist.as_slice())
                            .map(|(_, v)| u32::from_be_bytes(v.as_ref().try_into().unwrap()))
                            .collect();
                        for tid in found_matches { candidates.push((tid, 350.0)); }
                    } else {
                        let key = [label_id.to_be_bytes(), w1.to_be_bytes(), w2.to_be_bytes(), w3.to_be_bytes()].concat();
                        if let Some(bid) = links.get(&key).unwrap() {
                            candidates.push((u32::from_be_bytes(bid.as_ref().try_into().unwrap()), 350.0));
                        }
                    }
                }

                // HDC Semantic Momentum with Complexity Bias
                let mut context_vecs = Vec::new();
                for (pos, h_id) in history.iter().rev().take(5).enumerate() {
                    if let Some(v_bytes) = vectors.get(&h_id.to_be_bytes()).unwrap() {
                        context_vecs.push(shift_left(&bytes_to_bv(&v_bytes), pos));
                    }
                }

                if !context_vecs.is_empty() && candidates.len() < 10 {
                    let momentum = bundle(&context_vecs);
                    for item in links.iter().filter_map(Result::ok).take(8000) {
                        if label_id == 0 || item.0[0..4] == label_id.to_be_bytes() {
                            let tid = u32::from_be_bytes(item.1.as_ref().try_into().unwrap());
                            if let Some(v_b) = vectors.get(&tid.to_be_bytes()).unwrap() {
                                let v = bytes_to_bv(&v_b);
                                let sim = 1.0 - ((momentum.clone() ^ &v).count_ones() as f32 / DIMENSIONS as f32);
                                let bias = if label_id != 0 {
                                    1.0 - ((centroid.clone() ^ &v).count_ones() as f32 / DIMENSIONS as f32)
                                } else { 1.0 };
                                candidates.push((tid, sim * bias * 2.5));
                            }
                        }
                    }
                }

                if candidates.is_empty() { break; }
                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                let pool_size = (candidates.len() as f32 * temp.max(0.01)).ceil() as usize;
                let chosen = candidates[..pool_size.min(candidates.len())].choose(&mut rand::thread_rng()).unwrap().0;

                // Infinite Loop detection
                if history.len() > 10 {
                    let tail = &history[history.len()-4..];
                    if history.windows(4).rev().skip(1).take(15).any(|w| w == tail) { break; }
                }

                history.push(chosen);
                if let Some(wb) = vocab.get(&chosen.to_be_bytes()).unwrap() {
                    let word = String::from_utf8_lossy(&wb);
                    print!("{} ", word);
                    io::stdout().flush()?;
                    if word.contains('.') && i > 30 { break; }
                }
            }
            println!();
        }

        Commands::Prune { threshold } => {
            let mut count = 0;
            for item in weights.iter().filter_map(Result::ok) {
                if u32::from_be_bytes(item.1.as_ref().try_into().unwrap()) < threshold {
                    let _ = links.remove(&item.0);
                    count += 1;
                }
            }
            println!("Pruned {} low-frequency chains.", count);
        }
    }
    Ok(())
}