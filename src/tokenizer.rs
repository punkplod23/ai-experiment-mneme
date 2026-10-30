use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub const PAD_TOKEN_ID: u32 = 0;
pub const UNK_TOKEN_ID: u32 = 1;
pub const BOS_TOKEN_ID: u32 = 2;
pub const EOS_TOKEN_ID: u32 = 3;
pub const SYSTEM_START_ID: u32 = 4;
pub const SYSTEM_END_ID: u32 = 5;
pub const USER_START_ID: u32 = 6;
pub const USER_END_ID: u32 = 7;
pub const ASSISTANT_START_ID: u32 = 8;
pub const ASSISTANT_END_ID: u32 = 9;

pub const SPECIAL_TOKENS_COUNT: u32 = 10;
pub const BYTE_OFFSET: u32 = SPECIAL_TOKENS_COUNT; // 10
pub const BASE_VOCAB_SIZE: u32 = BYTE_OFFSET + 256; // 266

pub const SPECIAL_STRINGS: &[(&str, u32)] = &[
    ("<pad>", PAD_TOKEN_ID),
    ("<unk>", UNK_TOKEN_ID),
    ("<bos>", BOS_TOKEN_ID),
    ("<eos>", EOS_TOKEN_ID),
    ("<system>", SYSTEM_START_ID),
    ("</system>", SYSTEM_END_ID),
    ("<user>", USER_START_ID),
    ("</user>", USER_END_ID),
    ("<assistant>", ASSISTANT_START_ID),
    ("</assistant>", ASSISTANT_END_ID),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRule {
    pub pair: (u32, u32),
    pub new_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub vocab_size: usize,
    pub merges: Vec<MergeRule>,
    #[serde(skip)]
    merge_ranks: HashMap<(u32, u32), (u32, u32)>, // pair -> (rank, new_id)
    #[serde(skip)]
    id_to_bytes: HashMap<u32, Vec<u8>>,
}

/// Pre-tokenize text into word chunks while preserving leading whitespace, punctuation, and newlines.
/// Equivalent to the GPT-2 / LLaMA pre-tokenization strategy.
pub fn pre_tokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        if ch == '\n' || ch == '\r' || ch == '\t' {
            words.push(ch.to_string());
            i += 1;
        } else if ch.is_whitespace() {
            // Group whitespace with following alphanumeric characters
            let mut s = String::new();
            s.push(ch);
            i += 1;
            while i < chars.len() && chars[i].is_alphanumeric() {
                s.push(chars[i]);
                i += 1;
            }
            words.push(s);
        } else if ch.is_alphanumeric() {
            // Alphanumeric word without leading space (e.g. at start of line)
            let mut s = String::new();
            while i < chars.len() && chars[i].is_alphanumeric() {
                s.push(chars[i]);
                i += 1;
            }
            words.push(s);
        } else {
            // Punctuation or special symbol
            words.push(ch.to_string());
            i += 1;
        }
    }

    words
}

impl Tokenizer {
    pub fn new_base(target_vocab_size: usize) -> Self {
        let mut tok = Self {
            vocab_size: target_vocab_size,
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
            id_to_bytes: HashMap::new(),
        };
        tok.rebuild_cache();
        tok
    }

    pub fn rebuild_cache(&mut self) {
        self.merge_ranks.clear();
        self.id_to_bytes.clear();

        // Single byte tokens
        for b in 0u8..=255 {
            let id = BYTE_OFFSET + (b as u32);
            self.id_to_bytes.insert(id, vec![b]);
        }

        // Apply merges in order
        for (rank, rule) in self.merges.iter().enumerate() {
            self.merge_ranks.insert(rule.pair, (rank as u32, rule.new_id));
            let b1 = self.id_to_bytes.get(&rule.pair.0).cloned().unwrap_or_default();
            let b2 = self.id_to_bytes.get(&rule.pair.1).cloned().unwrap_or_default();
            let mut combined = b1;
            combined.extend_from_slice(&b2);
            self.id_to_bytes.insert(rule.new_id, combined);
        }
    }

    /// Train a BPE vocabulary on the given text corpus up to target_vocab_size.
    pub fn train(corpus: &str, target_vocab_size: usize) -> Self {
        let target_merges = target_vocab_size.saturating_sub(BASE_VOCAB_SIZE as usize);
        let mut merges = Vec::with_capacity(target_merges);

        // Pre-tokenize using whitespace-preserving chunks
        let word_chunks = pre_tokenize(corpus);
        let mut words: Vec<Vec<u32>> = word_chunks
            .into_iter()
            .map(|chunk| chunk.bytes().map(|b| BYTE_OFFSET + (b as u32)).collect())
            .filter(|w: &Vec<u32>| !w.is_empty())
            .collect();

        let mut next_id = BASE_VOCAB_SIZE;
        for _ in 0..target_merges {
            // Count pair frequencies
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for word in &words {
                if word.len() < 2 {
                    continue;
                }
                for window in word.windows(2) {
                    *pair_counts.entry((window[0], window[1])).or_insert(0) += 1;
                }
            }

            // Find pair with max frequency
            let best_pair = pair_counts.into_iter().max_by_key(|&(_, count)| count);
            match best_pair {
                Some((pair, count)) if count >= 2 => {
                    let rule = MergeRule { pair, new_id: next_id };
                    merges.push(rule);

                    // Update all words in-place with the new token
                    for word in &mut words {
                        if word.len() < 2 {
                            continue;
                        }
                        let mut new_word = Vec::with_capacity(word.len());
                        let mut i = 0;
                        while i < word.len() {
                            if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
                                new_word.push(next_id);
                                i += 2;
                            } else {
                                new_word.push(word[i]);
                                i += 1;
                            }
                        }
                        *word = new_word;
                    }

                    next_id += 1;
                }
                _ => break,
            }
        }

        let mut tok = Self {
            vocab_size: next_id as usize,
            merges,
            merge_ranks: HashMap::new(),
            id_to_bytes: HashMap::new(),
        };
        tok.rebuild_cache();
        tok
    }

    /// Encode arbitrary text into token IDs, recognizing special tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut cursor = 0;

        while cursor < text.len() {
            // Check for special token match at cursor
            let mut matched_special = None;
            for &(special_str, special_id) in SPECIAL_STRINGS {
                if text[cursor..].starts_with(special_str) {
                    matched_special = Some((special_str.len(), special_id));
                    break;
                }
            }

            if let Some((len, special_id)) = matched_special {
                tokens.push(special_id);
                cursor += len;
                continue;
            }

            // Find end of normal segment (until next special token or end)
            let mut end = text.len();
            for &(special_str, _) in SPECIAL_STRINGS {
                if let Some(pos) = text[cursor..].find(special_str) {
                    if pos > 0 && cursor + pos < end {
                        end = cursor + pos;
                    }
                }
            }

            let segment = &text[cursor..end];
            cursor = end;

            if segment.is_empty() {
                continue;
            }

            // Pre-tokenize segment into word chunks
            let chunks = pre_tokenize(segment);
            for chunk in chunks {
                let seg_bytes = chunk.as_bytes();
                if seg_bytes.is_empty() {
                    continue;
                }

                let mut word_tokens: Vec<u32> = seg_bytes
                    .iter()
                    .map(|&b| BYTE_OFFSET + (b as u32))
                    .collect();

                // Apply BPE merges iteratively within chunk
                if word_tokens.len() >= 2 && !self.merge_ranks.is_empty() {
                    loop {
                        let mut min_rank = u32::MAX;
                        let mut best_idx = None;

                        for i in 0..word_tokens.len() - 1 {
                            let pair = (word_tokens[i], word_tokens[i + 1]);
                            if let Some(&(rank, _)) = self.merge_ranks.get(&pair) {
                                if rank < min_rank {
                                    min_rank = rank;
                                    best_idx = Some(i);
                                }
                            }
                        }

                        if let Some(idx) = best_idx {
                            let pair = (word_tokens[idx], word_tokens[idx + 1]);
                            let new_id = self.merge_ranks.get(&pair).unwrap().1;
                            word_tokens[idx] = new_id;
                            word_tokens.remove(idx + 1);
                        } else {
                            break;
                        }
                    }
                }

                tokens.extend(word_tokens);
            }
        }

        tokens
    }

    /// Decode token IDs back to a UTF-8 string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();

        for &tok in tokens {
            let mut found_special = false;
            for &(special_str, special_id) in SPECIAL_STRINGS {
                if tok == special_id {
                    bytes.extend_from_slice(special_str.as_bytes());
                    found_special = true;
                    break;
                }
            }

            if !found_special {
                if let Some(b) = self.id_to_bytes.get(&tok) {
                    bytes.extend_from_slice(b);
                } else if tok >= BYTE_OFFSET && tok < BASE_VOCAB_SIZE {
                    bytes.push((tok - BYTE_OFFSET) as u8);
                }
            }
        }

        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Save tokenizer vocabulary and merges to JSON.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize tokenizer")?;
        fs::write(path, json).context("Failed to write tokenizer file")?;
        Ok(())
    }

    /// Load tokenizer vocabulary and merges from JSON.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path).context("Failed to read tokenizer file")?;
        let mut tok: Self = serde_json::from_str(&json).context("Failed to parse tokenizer JSON")?;
        tok.rebuild_cache();
        Ok(tok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_byte_fallback() {
        let tok = Tokenizer::new_base(BASE_VOCAB_SIZE as usize);
        let sample = "Hello, world! 1234 🚀";
        let encoded = tok.encode(sample);
        let decoded = tok.decode(&encoded);
        assert_eq!(sample, decoded);
    }

    #[test]
    fn test_tokenizer_bpe_training() {
        let corpus = "the quick brown fox jumps over the lazy dog. the quick brown fox.";
        let tok = Tokenizer::train(corpus, 350);
        assert!(tok.vocab_size > BASE_VOCAB_SIZE as usize);
        assert!(!tok.merges.is_empty());

        let encoded = tok.encode(corpus);
        let decoded = tok.decode(&encoded);
        assert_eq!(corpus, decoded);
    }

    #[test]
    fn test_tokenizer_preserves_spaces() {
        let corpus = "Isaac Newton was born in England. Gottfried Wilhelm Leibniz was born in Germany.";
        let tok = Tokenizer::train(corpus, 350);
        let encoded = tok.encode(corpus);
        let decoded = tok.decode(&encoded);
        assert_eq!(corpus, decoded);
    }

    #[test]
    fn test_tokenizer_special_tokens() {
        let tok = Tokenizer::new_base(BASE_VOCAB_SIZE as usize);
        let text = "<user>Hello assistant</user><assistant>Greetings!</assistant>";
        let encoded = tok.encode(text);
        assert_eq!(encoded[0], USER_START_ID);
        assert!(encoded.contains(&USER_END_ID));
        assert!(encoded.contains(&ASSISTANT_START_ID));
        assert_eq!(*encoded.last().unwrap(), ASSISTANT_END_ID);

        let decoded = tok.decode(&encoded);
        assert_eq!(text, decoded);
    }
}
