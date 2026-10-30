pub mod memory;
pub mod model;
pub mod tokenizer;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use rand::seq::SliceRandom;
use rand::Rng;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub use memory::ConversationManager;
pub use model::{Mamba3Sovereign, MambaConfig};
pub use tokenizer::Tokenizer;

pub const SAVE_FILE: &str = "mamba3_sov.safetensors";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const CONFIG_FILE: &str = "mamba_config.json";

#[derive(Parser, Debug)]
#[command(
    name = "mamba3_sov",
    about = "Mneme: Mamba 3 Sovereign Selective State-Space Language Model in Pure Rust"
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Ingest and train on a text file or directory of text files comprehensively
    Ingest {
        /// Path to a text file or directory of books
        path: String,
        /// Number of training epochs
        #[arg(short, long, default_value_t = 10)]
        epochs: usize,
        /// Mini-batch size
        #[arg(short, long, default_value_t = 8)]
        batch_size: usize,
        /// Learning rate
        #[arg(long, default_value_t = 5e-4)]
        lr: f64,
        /// Maximum number of batches per epoch (defaults to all batches in corpus)
        #[arg(long)]
        max_batches: Option<usize>,
        /// Reset model weights and start training from scratch
        #[arg(short, long)]
        reset: bool,
        /// Force CPU for ingestion even if CUDA is available
        #[arg(long)]
        cpu: bool,
    },
    /// Chat with the model (interactive multi-turn conversation or one-shot prompt)
    Chat {
        /// Optional one-shot prompt (if omitted, launches interactive conversational REPL)
        #[arg(default_value = "")]
        prompt: String,
        /// Seed text alias for prompt
        #[arg(short, long)]
        seed: Option<String>,
        /// Sampling temperature (lower = deterministic, higher = creative)
        #[arg(short, long, default_value_t = 0.7)]
        temp: f32,
        /// Maximum tokens to generate
        #[arg(short, long, alias = "length", default_value_t = 150)]
        len: usize,
        /// Custom system prompt for the sovereign intelligence
        #[arg(long)]
        system: Option<String>,
        /// Top-K sampling pool (default 40)
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// Legacy routing label id (retained for script compatibility)
        #[arg(long)]
        label_id: Option<usize>,
        /// Force CPU for chat even if CUDA is available
        #[arg(long)]
        cpu: bool,
    },
}

/// Helper to load or train the BPE subword tokenizer.
pub fn get_or_train_tokenizer(corpus_sample_path: Option<&Path>) -> Result<Tokenizer> {
    if Path::new(TOKENIZER_FILE).exists() {
        println!("Loading subword tokenizer from '{}'...", TOKENIZER_FILE);
        return Tokenizer::load(TOKENIZER_FILE);
    }

    println!("Tokenizer not found. Training BPE Subword Tokenizer (vocab size: 2048)...");
    let mut sample_text = String::new();

    // Prioritize gathering text from books directory if available
    let books_dir = Path::new("books");
    if books_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(books_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().is_some_and(|ext| ext == "txt") {
                    let name = path.file_name().unwrap_or_default().to_string_lossy();
                    if name != "corpus.txt" {
                        if let Ok(txt) = fs::read_to_string(&path) {
                            let take_len = txt.len().min(15_000);
                            sample_text.push_str(&txt[..take_len]);
                            sample_text.push('\n');
                            if sample_text.len() >= 60_000 {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    if sample_text.is_empty() {
        if let Some(p) = corpus_sample_path {
            if let Ok(txt) = fs::read_to_string(p) {
                sample_text = txt;
            }
        }
    }

    if sample_text.is_empty() {
        sample_text = "Default sovereign intelligence corpus on philosophy, logic, and computation.".to_string();
    }

    let tok = Tokenizer::train(&sample_text, 2048);
    tok.save(TOKENIZER_FILE)?;
    println!(
        "BPE Tokenizer trained successfully (Vocab: {} tokens) and saved to '{}'.",
        tok.vocab_size, TOKENIZER_FILE
    );
    Ok(tok)
}

/// Helper to load or save Mamba model configuration.
pub fn get_or_create_config(vocab_size: usize) -> Result<MambaConfig> {
    if Path::new(CONFIG_FILE).exists() {
        let json = fs::read_to_string(CONFIG_FILE)?;
        if let Ok(mut cfg) = serde_json::from_str::<MambaConfig>(&json) {
            cfg.vocab_size = vocab_size;
            return Ok(cfg);
        }
    }

    let cfg = MambaConfig {
        d_model: 256,
        d_inner: 512,
        d_state: 32,
        d_conv: 4,
        dt_rank: 16,
        num_layers: 4,
        vocab_size,
        chunk_size: 256,
    };
    let json = serde_json::to_string_pretty(&cfg)?;
    fs::write(CONFIG_FILE, json)?;
    Ok(cfg)
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Commands::Ingest {
            path,
            epochs,
            batch_size,
            lr,
            max_batches,
            reset,
            cpu,
        } => {
            let device = if cpu {
                Device::Cpu
            } else {
                Device::cuda_if_available(0)?
            };
            println!("[Sovereign Hardware: {:?}]", device);

            let p = Path::new(&path);
            let files: Vec<PathBuf> = if p.is_dir() {
                let mut entries = Vec::new();
                for entry in fs::read_dir(p)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.is_file() && path.extension().is_some_and(|ext| ext == "txt") {
                        let name = path.file_name().unwrap_or_default().to_string_lossy();
                        if name != "corpus.txt" {
                            entries.push(path);
                        }
                    }
                }
                entries.sort();
                entries
            } else {
                vec![p.to_path_buf()]
            };

            if files.is_empty() {
                bail!("No valid text files found in '{}'.", path);
            }

            // 1. Prepare Subword Tokenizer
            let tokenizer = get_or_train_tokenizer(Some(p))?;
            let cfg = get_or_create_config(tokenizer.vocab_size)?;

            // 2. Initialize Mamba Model
            let mut varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let model = Mamba3Sovereign::new(cfg.clone(), vb)?;

            let save_path = Path::new(SAVE_FILE);
            if save_path.exists() && !reset {
                match varmap.load(SAVE_FILE) {
                    Ok(_) => println!(
                        "Loaded existing {}-layer Mamba weights from '{}'",
                        cfg.num_layers, SAVE_FILE
                    ),
                    Err(e) => println!(
                        "Warning: Could not load '{}': {}. Initializing fresh weights.",
                        SAVE_FILE, e
                    ),
                }
            } else {
                println!(
                    "Initialized fresh {}-layer Mamba Sovereign model (d_model: {}, d_inner: {}, d_state: {}, vocab: {}).",
                    cfg.num_layers, cfg.d_model, cfg.d_inner, cfg.d_state, cfg.vocab_size
                );
            }

            // 3. Optimizer
            let mut opt = AdamW::new(
                varmap.all_vars(),
                ParamsAdamW {
                    lr,
                    ..Default::default()
                },
            )?;

            println!(
                "\n=== Starting Sovereign Comprehensive Ingestion ==="
            );
            println!(
                "Queued {} file(s) | Epochs: {} | Batch Size: {} | Context Chunk: {} tokens",
                files.len(),
                epochs,
                batch_size,
                cfg.chunk_size
            );

            let chunk_len = cfg.chunk_size;
            let sample_len = chunk_len + 1; // input + target next token

            for (file_idx, file_path) in files.iter().enumerate() {
                let file_name = file_path.file_name().unwrap_or_default().to_string_lossy();
                let text = match fs::read_to_string(file_path) {
                    Ok(t) => t,
                    Err(e) => {
                        println!("Skipping '{}': {}", file_name, e);
                        continue;
                    }
                };

                let mut tokens = tokenizer.encode(&text);
                if tokens.len() < 4 {
                    println!("Skipping '{}': file too small", file_name);
                    continue;
                }

                // If file is smaller than chunk size, repeat it to fill at least sample_len
                while tokens.len() <= sample_len {
                    tokens.extend_from_slice(&tokenizer.encode(&text));
                }

                // Sliding window stride: smaller stride on small datasets for rich sample diversity
                let stride = if tokens.len() < chunk_len * 2 {
                    (chunk_len / 4).max(16)
                } else {
                    chunk_len / 2
                };

                let mut chunk_starts = Vec::new();
                for s in (0..tokens.len().saturating_sub(sample_len)).step_by(stride) {
                    chunk_starts.push(s);
                }
                if chunk_starts.is_empty() {
                    chunk_starts.push(0);
                }

                let num_chunks = chunk_starts.len();
                let total_batches = (num_chunks + batch_size - 1) / batch_size;
                let batches_to_run = if let Some(max_b) = max_batches {
                    total_batches.min(max_b)
                } else {
                    total_batches
                };

                println!(
                    "\n[{}/{}] Ingesting '{}' ({} KB -> {} subword tokens, {} sliding chunks, {} batches)...",
                    file_idx + 1,
                    files.len(),
                    file_name,
                    text.len() / 1024,
                    tokens.len(),
                    num_chunks,
                    batches_to_run
                );

                let mut rng = rand::rng();

                for epoch in 1..=epochs {
                    let start_epoch = Instant::now();
                    let mut epoch_loss = 0.0f32;
                    let mut steps = 0usize;
                    let mut tokens_processed = 0usize;

                    let mut epoch_starts = chunk_starts.clone();
                    epoch_starts.shuffle(&mut rng);

                    for b_idx in 0..batches_to_run {
                        let mut batch_inputs = Vec::with_capacity(batch_size * chunk_len);
                        let mut batch_targets = Vec::with_capacity(batch_size * chunk_len);
                        let mut actual_batch = 0;

                        for i in 0..batch_size {
                            let chunk_id = b_idx * batch_size + i;
                            if chunk_id < epoch_starts.len() {
                                let start_tok = epoch_starts[chunk_id];
                                if start_tok + sample_len <= tokens.len() {
                                    batch_inputs
                                        .extend_from_slice(&tokens[start_tok..start_tok + chunk_len]);
                                    batch_targets.extend_from_slice(
                                        &tokens[start_tok + 1..start_tok + sample_len],
                                    );
                                    actual_batch += 1;
                                }
                            }
                        }

                        if actual_batch == 0 {
                            break;
                        }

                        let input_t = Tensor::from_slice(
                            &batch_inputs,
                            (actual_batch, chunk_len),
                            &device,
                        )?;
                        let target_t = Tensor::from_slice(
                            &batch_targets,
                            (actual_batch * chunk_len,),
                            &device,
                        )?;

                        let logits = model
                            .forward(&input_t)?
                            .reshape((actual_batch * chunk_len, cfg.vocab_size))?;

                        let loss = candle_nn::loss::cross_entropy(&logits, &target_t)?;
                        opt.backward_step(&loss)?;

                        let loss_val = loss.to_vec0::<f32>()?;
                        epoch_loss += loss_val;
                        steps += 1;
                        tokens_processed += actual_batch * chunk_len;
                    }

                    let elapsed = start_epoch.elapsed().as_secs_f32().max(0.001);
                    let avg_loss = if steps > 0 {
                        epoch_loss / steps as f32
                    } else {
                        0.0
                    };
                    let tps = tokens_processed as f32 / elapsed;

                    println!(
                        "  Epoch {:>2}/{:<2} | Loss: {:.4} | TPS: {:>5.0} | Elapsed: {:.1}s",
                        epoch, epochs, avg_loss, tps, elapsed
                    );

                    if epoch % 10 == 0 || epoch == epochs {
                        varmap.save(SAVE_FILE)?;
                    }
                }

                // Persist updated weights
                varmap.save(SAVE_FILE)?;
            }

            println!(
                "\nIngestion Complete. Sovereign Mamba weights successfully saved to '{}'.",
                SAVE_FILE
            );
        }

        Commands::Chat {
            prompt,
            seed,
            temp,
            len,
            system,
            top_k,
            label_id,
            cpu,
        } => {
            let device = if cpu {
                Device::Cpu
            } else {
                Device::cuda_if_available(0)?
            };
            println!("[Sovereign Hardware: {:?}]", device);

            if !Path::new(SAVE_FILE).exists() {
                bail!(
                    "Model weights '{}' not found. Please train first via 'mamba3_sov ingest <path>'.",
                    SAVE_FILE
                );
            }

            let tokenizer = Tokenizer::load(TOKENIZER_FILE).context(
                "Tokenizer 'tokenizer.json' not found. Ingest text first to generate vocabulary.",
            )?;
            let cfg = get_or_create_config(tokenizer.vocab_size)?;

            let mut varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let model = Mamba3Sovereign::new(cfg.clone(), vb)?;
            varmap.load(SAVE_FILE)?;

            if let Some(lid) = label_id {
                println!("[Routing Context: Label {}]", lid);
            }

            let one_shot_prompt = if !prompt.is_empty() {
                Some(prompt)
            } else {
                seed
            };

            if let Some(user_input) = one_shot_prompt {
                // One-shot completion mode (maintains script/makefile compatibility)
                run_one_shot(&model, &tokenizer, &user_input, temp, len, top_k, &device)?;
            } else {
                // Interactive Conversational REPL mode (multi-turn memory)
                run_interactive_chat(&model, &tokenizer, system.as_deref(), temp, len, top_k, &device)?;
            }
        }
    }

    Ok(())
}

/// One-shot prompt completion (for CLI commands and makefile scripts).
fn run_one_shot(
    model: &Mamba3Sovereign,
    tokenizer: &Tokenizer,
    prompt: &str,
    temp: f32,
    max_len: usize,
    top_k: usize,
    device: &Device,
) -> Result<()> {
    println!("\n[Sovereign Response]:");
    print!("{}", prompt);
    io::stdout().flush()?;

    let mut tokens = tokenizer.encode(prompt);
    if tokens.is_empty() {
        tokens.push(tokenizer::BOS_TOKEN_ID);
    }

    let mut rng = rand::rng();

    for _ in 0..max_len {
        let window_len = tokens.len().min(model.cfg.chunk_size);
        let start = tokens.len() - window_len;
        let context = &tokens[start..];

        let input_t = Tensor::from_slice(context, (1, window_len), device)?;
        let logits = model.forward(&input_t)?;
        let last_logits = logits.i((0, window_len - 1))?;
        let mut logits_vec = last_logits.to_vec1::<f32>()?;

        // Repetition penalty for recently seen tokens
        let search_start = tokens.len().saturating_sub(16);
        for &tok in &tokens[search_start..] {
            if (tok as usize) < logits_vec.len() {
                logits_vec[tok as usize] -= 1.8;
            }
        }

        // Penalize unprintable control bytes (0..31 except \n, \t, \r)
        for tok_id in tokenizer::BYTE_OFFSET as usize..(tokenizer::BYTE_OFFSET + 32) as usize {
            if tok_id < logits_vec.len() {
                let b = (tok_id - tokenizer::BYTE_OFFSET as usize) as u8;
                if b != b'\n' && b != b'\t' && b != b'\r' {
                    logits_vec[tok_id] -= 50.0;
                }
            }
        }

        let next_token = sample_token(&logits_vec, temp, top_k, &mut rng);
        if next_token == tokenizer::EOS_TOKEN_ID || next_token == tokenizer::ASSISTANT_END_ID {
            break;
        }

        let piece = tokenizer.decode(&[next_token]);
        print!("{}", piece);
        io::stdout().flush()?;
        tokens.push(next_token);
    }

    println!("\n");
    Ok(())
}

/// Interactive Multi-Turn Conversational REPL with Memory.
fn run_interactive_chat(
    model: &Mamba3Sovereign,
    tokenizer: &Tokenizer,
    custom_system_prompt: Option<&str>,
    mut temp: f32,
    max_len: usize,
    top_k: usize,
    device: &Device,
) -> Result<()> {
    let mut memory = ConversationManager::new(
        custom_system_prompt.unwrap_or(
            "You are Mneme, a sovereign Mamba intelligence. You are versed in philosophy, mathematics, and natural science. Answer thoughtfully.",
        ),
        model.cfg.chunk_size * 2,
    );

    println!();
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                   Mneme: Mamba 3 Sovereign Chat Console                ║");
    println!("║          Linear-Time State Space Intelligence (Pure Rust / Candle)      ║");
    println!("║                                                                        ║");
    println!("║   Commands:                                                            ║");
    println!("║     /exit or /quit  - Leave the chat session                           ║");
    println!("║     /reset or /clear - Wipe conversation memory                        ║");
    println!("║     /history         - View active conversation dialogue history       ║");
    println!("║     /system <prompt> - Update system personality on the fly            ║");
    println!("║     /temp <value>    - Adjust sampling temperature (e.g. /temp 0.8)    ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!("Model: {} layers | d_model: {} | d_inner: {} | Vocab: {} subwords\n",
        model.cfg.num_layers, model.cfg.d_model, model.cfg.d_inner, model.cfg.vocab_size);

    let stdin = io::stdin();
    let mut rng = rand::rng();

    loop {
        print!("User > ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        // Handle slash commands
        if input == "/exit" || input == "/quit" {
            println!("Farewell.");
            break;
        } else if input == "/reset" || input == "/clear" {
            memory.clear();
            println!("[Memory reset. Conversation history cleared.]\n");
            continue;
        } else if input == "/history" {
            println!("\n--- Stored Conversation Turns ({} total) ---", memory.history.len());
            for msg in &memory.history {
                println!("[{:?}]: {}", msg.role, msg.content);
            }
            println!("--------------------------------------------\n");
            continue;
        } else if input.starts_with("/temp ") {
            if let Ok(new_t) = input[6..].trim().parse::<f32>() {
                temp = new_t.clamp(0.01, 2.0);
                println!("[Temperature set to {:.2}]\n", temp);
            } else {
                println!("[Invalid temperature value]\n");
            }
            continue;
        } else if input.starts_with("/system ") {
            let new_sys = input[8..].trim();
            memory.system_prompt = new_sys.to_string();
            println!("[System prompt updated]\n");
            continue;
        }

        // Add user turn to conversational memory
        memory.add_user_message(input);

        // Format prompt using structured dialogue tokens
        let mut prompt_tokens = memory.format_prompt(tokenizer);

        print!("Mneme > ");
        io::stdout().flush()?;

        let mut response_tokens = Vec::new();

        for _ in 0..max_len {
            let window_len = prompt_tokens.len().min(model.cfg.chunk_size);
            let start = prompt_tokens.len() - window_len;
            let context = &prompt_tokens[start..];

            let input_t = Tensor::from_slice(context, (1, window_len), device)?;
            let logits = model.forward(&input_t)?;
            let last_logits = logits.i((0, window_len - 1))?;
            let mut logits_vec = last_logits.to_vec1::<f32>()?;

            // Repetition penalty for recently emitted tokens
            let search_start = prompt_tokens.len().saturating_sub(20);
            for &tok in &prompt_tokens[search_start..] {
                if (tok as usize) < logits_vec.len() {
                    logits_vec[tok as usize] -= 1.6;
                }
            }

            // Penalize unprintable control bytes (0..31 except \n, \t, \r)
            for tok_id in tokenizer::BYTE_OFFSET as usize..(tokenizer::BYTE_OFFSET + 32) as usize {
                if tok_id < logits_vec.len() {
                    let b = (tok_id - tokenizer::BYTE_OFFSET as usize) as u8;
                    if b != b'\n' && b != b'\t' && b != b'\r' {
                        logits_vec[tok_id] -= 50.0;
                    }
                }
            }

            let next_token = sample_token(&logits_vec, temp, top_k, &mut rng);

            // Stop generation at end-of-turn or end-of-sequence
            if next_token == tokenizer::EOS_TOKEN_ID || next_token == tokenizer::ASSISTANT_END_ID {
                break;
            }

            let token_str = tokenizer.decode(&[next_token]);
            print!("{}", token_str);
            io::stdout().flush()?;

            prompt_tokens.push(next_token);
            response_tokens.push(next_token);
        }

        println!("\n");
        let assistant_reply = tokenizer.decode(&response_tokens);
        memory.add_assistant_message(&assistant_reply);
    }

    Ok(())
}

/// Temperature and Top-K sampling function.
fn sample_token(logits: &[f32], temp: f32, top_k: usize, rng: &mut impl Rng) -> u32 {
    if temp <= 0.01 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    let inv_temp = 1.0 / temp;
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut exps: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, ((l - max_l) * inv_temp).exp()))
        .collect();

    // Top-K Filtering
    exps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let pool_size = top_k.min(exps.len()).max(1);
    let top_pool = &exps[..pool_size];

    let sum: f32 = top_pool.iter().map(|(_, p)| p).sum();
    let r = rng.random::<f32>() * sum;
    let mut cumsum = 0.0f32;
    for &(id, p) in top_pool {
        cumsum += p;
        if r <= cumsum {
            return id;
        }
    }
    top_pool[0].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_token_argmax() {
        let mut rng = rand::rng();
        let logits = vec![0.1, 0.9, 0.3, 0.4];
        let chosen = sample_token(&logits, 0.0, 4, &mut rng);
        assert_eq!(chosen, 1);
    }
}
