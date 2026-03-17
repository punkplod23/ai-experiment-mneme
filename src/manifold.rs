use serde::{Serialize, Deserialize};
use std::collections::{HashSet, HashMap};
use blake3;

pub const DIMENSIONS: usize = 65536;
const SATURATION_THRESHOLD: f32 = 0.30; 
const COOLING_FACTOR: f32 = 0.30; 
const BITS_PER_TOKEN: usize = 16; 

#[derive(Serialize, Deserialize)]
pub struct LeibnizManifold {
    pub weights: Vec<f32>,
    pub dictionary: HashSet<String>,
    pub frequency_map: HashMap<String, u32>,
}

impl LeibnizManifold {
    pub fn new() -> Self {
        Self {
            weights: vec![0.0; DIMENSIONS],
            dictionary: HashSet::new(),
            frequency_map: HashMap::new(),
        }
    }

    pub fn get_saturation(&self) -> f32 {
        let active = self.weights.iter().filter(|&&w| w > 0.1).count();
        active as f32 / DIMENSIONS as f32
    }

    pub fn normalize(&mut self) {
        println!("\n[AUTO-NORM] Cooling manifold...");
        for w in self.weights.iter_mut() {
            *w *= COOLING_FACTOR;
            if *w < 0.05 { *w = 0.0; } 
        }
    }

    pub fn ingest_token(&mut self, token: &str) {
        if self.get_saturation() > SATURATION_THRESHOLD {
            self.normalize();
        }

        let count = self.frequency_map.entry(token.to_string()).or_insert(0);
        *count += 1;

        // Dampen based on frequency
        let weight_boost = 1.0 / (*count as f32).log2().max(1.0);

        let hash = blake3::hash(token.as_bytes());
        let bytes = hash.as_bytes();

        for i in 0..BITS_PER_TOKEN {
            let b1 = bytes[i % 16] as usize;
            let b2 = bytes[(i + 1) % 32] as usize;
            let idx = (b1 << 8 | b2) % DIMENSIONS;
            self.weights[idx] += weight_boost;
        }
        self.dictionary.insert(token.to_string());
    }

    pub fn query(&self, text: &str) -> Vec<(String, f32)> {
        // 1. Build the Query Vector (The Probe)
        let mut probe = vec![0.0; DIMENSIONS];
        for token in text.split_whitespace() {
            let clean = token.to_lowercase().replace(|c: char| !c.is_alphanumeric(), "");
            let hash = blake3::hash(clean.as_bytes());
            let bytes = hash.as_bytes();
            for i in 0..BITS_PER_TOKEN {
                let idx = ((bytes[i % 16] as usize) << 8 | (bytes[(i + 1) % 32] as usize)) % DIMENSIONS;
                probe[idx] = 1.0;
            }
        }

        // 2. Score using a dot-product normalized by word frequency
        let mut results: Vec<(String, f32)> = self.dictionary.iter().map(|word| {
            let mut dot_product = 0.0;
            let hash = blake3::hash(word.as_bytes());
            let bytes = hash.as_bytes();
            
            for i in 0..BITS_PER_TOKEN {
                let idx = ((bytes[i % 16] as usize) << 8 | (bytes[(i + 1) % 32] as usize)) % DIMENSIONS;
                dot_product += self.weights[idx] * probe[idx];
            }
            
            // Frequency adjustment: We want high overlap but we penalize high frequency
            // and ultra-low frequency outliers by using a square-root of log.
            let freq = *self.frequency_map.get(word).unwrap_or(&1) as f32;
            let final_score = dot_product / freq.log2().max(1.1).sqrt();
            
            (word.clone(), final_score)
        }).collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().take(5).collect()
    }
}