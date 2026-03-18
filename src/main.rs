use clap::{Parser, Subcommand};
use half::f16;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

const DIM: usize = 512;
const HIDDEN_DIM: usize = 2048;
const NUM_LAYERS: usize = 4;
const WINDOW_SIZE: usize = 128; // The "Memory" length
const SAVE_FILE: &str = "sovereign_brain.bin";
const MAX_VOCAB: usize = 1024;
const LEARNING_RATE: f32 = 0.0004;
const WEIGHT_DECAY: f32 = 0.000001;
const GRAD_CLIP: f32 = 0.02;

// --- Attention & HDC Kernels ---
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// Simple Softmax for Attention Weights
fn attention_weights(query: &[f32], keys: &[Vec<f32>]) -> Vec<f32> {
    let mut scores: Vec<f32> = keys.iter().map(|k| dot(query, k) / (DIM as f32).sqrt()).collect();
    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp();
        sum += *s;
    }
    for s in scores.iter_mut() { *s /= sum; }
    scores
}

fn layer_norm(x: &mut [f32]) {
    let len = x.len() as f32;
    let mean = x.iter().sum::<f32>() / len;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / len;
    let inv_std = 1.0 / (var + 1e-6).sqrt();
    for v in x.iter_mut() { *v = ((*v - mean) * inv_std).clamp(-5.0, 5.0); }
}

// HDC Bind now incorporates the Attention context
fn hdc_attention_bind(current: &[f32], window: &[Vec<f32>]) -> Vec<f32> {
    if window.is_empty() { return current.to_vec(); }
    
    let weights = attention_weights(current, window);
    let mut context = vec![0.0; DIM];
    
    for (i, w) in weights.iter().enumerate() {
        for j in 0..DIM {
            context[j] += window[i][j] * w;
        }
    }
    
    let mut result = vec![0.0; DIM];
    for i in 0..DIM {
        result[i] = (context[(i + 1) % DIM] * 0.5) + current[i];
    }
    layer_norm(&mut result);
    result
}

#[derive(Serialize, Deserialize, Default)]
struct Tokenizer {
    vocab: HashMap<Vec<u8>, u32>,
    id_to_token: HashMap<u32, Vec<u8>>,
}

impl Tokenizer {
    fn train(&mut self, data: &[u8]) {
        self.vocab.clear(); self.id_to_token.clear();
        for i in 0..256 {
            let b = vec![i as u8];
            self.vocab.insert(b.clone(), i as u32);
            self.id_to_token.insert(i as u32, b);
        }
        let mut counts: HashMap<(u8, u8), usize> = HashMap::new();
        for win in data.windows(2) { *counts.entry((win[0], win[1])).or_insert(0) += 1; }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let mut next_id = 256;
        for ((a, b), _) in sorted.into_iter().take(MAX_VOCAB - 256) {
            let combo = vec![a, b];
            self.vocab.insert(combo.clone(), next_id);
            self.id_to_token.insert(next_id, combo);
            next_id += 1;
        }
    }
    fn encode(&self, data: &[u8]) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < data.len() {
            if i + 1 < data.len() {
                if let Some(&id) = self.vocab.get(&vec![data[i], data[i+1]]) {
                    tokens.push(id); i += 2; continue;
                }
            }
            tokens.push(data[i] as u32); i += 1;
        }
        tokens
    }
    fn decode(&self, id: u32) -> String {
        self.id_to_token.get(&id).map(|b| String::from_utf8_lossy(b).to_string()).unwrap_or_default()
    }
}

#[derive(Serialize, Deserialize)]
struct Layer { w_out: Vec<f16>, ff1: Vec<f16>, ff2: Vec<f16> }

impl Layer {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut x_n = x.to_vec();
        layer_norm(&mut x_n);
        let mut mid = (0..DIM).map(|r| dot(&dequant(&self.w_out[r*DIM..(r+1)*DIM]), &x_n)).collect::<Vec<_>>();
        for i in 0..DIM { mid[i] = (mid[i] + x[i]).clamp(-10.0, 10.0); }
        layer_norm(&mut mid);
        let mut hid = (0..HIDDEN_DIM).map(|r| dot(&dequant(&self.ff1[r*DIM..(r+1)*DIM]), &mid)).collect::<Vec<_>>();
        for v in &mut hid { if *v < 0.0 { *v = 0.0; } }
        let out = (0..DIM).map(|r| dot(&dequant(&self.ff2[r*HIDDEN_DIM..(r+1)*HIDDEN_DIM]), &hid)).collect::<Vec<_>>();
        mid.iter().zip(out).map(|(a, b)| (a + b).clamp(-10.0, 10.0)).collect()
    }
}

fn dequant(v: &[f16]) -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() }

#[derive(Serialize, Deserialize)]
struct SovereignBrain {
    tokenizer: Tokenizer,
    embeddings: Vec<f32>, 
    layers: Vec<Layer>,
    pub completed_epochs: usize,
}

impl SovereignBrain {
    fn train_overdrive(&mut self, tokens: &[u32], pb: &ProgressBar) -> f32 {
        let start = Instant::now();
        let num_tasks = 256; 
        let chunk_size = (tokens.len() / num_tasks).max(1);
        let ptr = self.embeddings.as_mut_ptr() as usize;
        let total_loss_atomic = Arc::new(AtomicU32::new(0));

        tokens.par_chunks(chunk_size).for_each(|chunk| {
            let mut window: Vec<Vec<f32>> = Vec::with_capacity(WINDOW_SIZE);
            let mut local_loss = 0.0;
            
            for &token_id in chunk {
                let i_idx = token_id as usize * DIM;
                
                unsafe {
                    let embs = std::slice::from_raw_parts_mut(ptr as *mut f32, MAX_VOCAB * DIM);
                    
                    // Attention-based Forward
                    let mut x = hdc_attention_bind(&embs[i_idx..i_idx+DIM], &window);
                    for l in &self.layers { x = l.forward(&x); }
                    
                    // Predict next token in stream (simple lookahead for training)
                    // In a real sliding window, we'd predict the next actual token
                    // but for Hogwild speed, we update the current embedding's relation to context
                    for i in 0..DIM {
                        let target = embs[i_idx + i];
                        let diff = (target - x[i]).clamp(-GRAD_CLIP, GRAD_CLIP);
                        local_loss += diff.powi(2);
                        embs[i_idx + i] = (target * (1.0 - WEIGHT_DECAY)) + (diff * LEARNING_RATE);
                    }
                    
                    // Update the rolling window
                    if window.len() >= WINDOW_SIZE { window.remove(0); }
                    window.push(x);
                }
            }
            pb.inc(chunk.len() as u64);
            total_loss_atomic.fetch_add((local_loss * 100.0) as u32, Ordering::Relaxed);
        });

        let duration = start.elapsed().as_secs_f32();
        pb.set_message(format!("TPS: {:.0}", tokens.len() as f32 / duration));
        (total_loss_atomic.load(Ordering::Relaxed) as f32 / 100.0) / tokens.len() as f32
    }
}

#[derive(Parser)] struct Cli { #[command(subcommand)] cmd: Commands }
#[derive(Subcommand)] enum Commands { 
    Ingest { path: String, #[arg(short, long, default_value_t = 20)] epochs: usize },
    Chat { seed: String, #[arg(short, long, default_value_t = 100)] len: usize, #[arg(short, long, default_value_t = 0.8)] temp: f32 }
}

fn main() -> io::Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(10).build_global().unwrap();
    let cli = Cli::parse();
    let mut rng = ChaCha8Rng::from_entropy();

    match cli.cmd {
        Commands::Ingest { path, epochs } => {
            let content = fs::read(&path)?;
            let mut brain = if let Ok(data) = fs::read(SAVE_FILE) {
                bincode::deserialize(&data).unwrap()
            } else {
                let s_layer = (6.0 / (DIM + DIM) as f32).sqrt();
                let s_ff = (6.0 / (DIM + HIDDEN_DIM) as f32).sqrt();
                let mut b = SovereignBrain {
                    tokenizer: Tokenizer::default(),
                    embeddings: (0..MAX_VOCAB * DIM).map(|_| rng.gen_range(-0.01..0.01)).collect(),
                    layers: (0..NUM_LAYERS).map(|_| Layer {
                        w_out: (0..DIM*DIM).map(|_| f16::from_f32(rng.gen_range(-s_layer..s_layer))).collect(),
                        ff1: (0..HIDDEN_DIM*DIM).map(|_| f16::from_f32(rng.gen_range(-s_ff..s_ff))).collect(),
                        ff2: (0..DIM*HIDDEN_DIM).map(|_| f16::from_f32(rng.gen_range(-s_ff..s_ff))).collect(),
                    }).collect(),
                    completed_epochs: 0,
                };
                b.tokenizer.train(&content); b
            };

            let tokens = brain.tokenizer.encode(&content);
            let pb = ProgressBar::new(tokens.len() as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap());

            for e in (brain.completed_epochs + 1)..=epochs {
                pb.reset();
                let loss = brain.train_overdrive(&tokens, &pb);
                brain.completed_epochs = e;
                fs::write(SAVE_FILE, bincode::serialize(&brain).unwrap()).unwrap();
            }
            pb.finish_with_message("Attention Ingestion Complete.");
            Ok(())
        }
        Commands::Chat { seed, len, temp } => {
            let brain: SovereignBrain = bincode::deserialize(&fs::read(SAVE_FILE)?).unwrap();
            let mut tokens = brain.tokenizer.encode(seed.as_bytes());
            let mut window: Vec<Vec<f32>> = Vec::new();
            
            println!("\n[Sovereign]: {}", seed);
            for _ in 0..len {
                let last = *tokens.last().unwrap() as usize;
                let mut x = hdc_attention_bind(&brain.embeddings[last*DIM..(last+1)*DIM], &window);
                for l in &brain.layers { x = l.forward(&x); }
                
                let mut logits: Vec<f32> = (0..MAX_VOCAB).map(|i| dot(&x, &brain.embeddings[i*DIM..(i+1)*DIM]) / temp).collect();
                let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0;
                for l in logits.iter_mut() { *l = (*l - max_logit).exp(); sum += *l; }
                let r = rand::thread_rng().gen_range(0.0..sum);
                let mut acc = 0.0;
                let next = logits.iter().enumerate().find_map(|(i, &p)| { acc += p; if acc >= r { Some(i as u32) } else { None } }).unwrap_or(0);
                
                tokens.push(next);
                print!("{}", brain.tokenizer.decode(next));
                io::stdout().flush().unwrap();
                
                if window.len() >= WINDOW_SIZE { window.remove(0); }
                window.push(x);
            }
            Ok(())
        }
    }
}