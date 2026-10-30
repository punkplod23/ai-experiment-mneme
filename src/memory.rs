use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationManager {
    pub system_prompt: String,
    pub history: Vec<ChatMessage>,
    pub max_context_tokens: usize,
}

impl Default for ConversationManager {
    fn default() -> Self {
        Self::new("You are Mneme, a sovereign Mamba intelligence. You are knowledgeable in philosophy, science, and reasoning. Respond thoughtfully and concisely.", 512)
    }
}

impl ConversationManager {
    pub fn new(system_prompt: &str, max_context_tokens: usize) -> Self {
        Self {
            system_prompt: system_prompt.to_string(),
            history: Vec::new(),
            max_context_tokens,
        }
    }

    pub fn add_user_message(&mut self, content: &str) {
        self.history.push(ChatMessage {
            role: Role::User,
            content: content.trim().to_string(),
        });
    }

    pub fn add_assistant_message(&mut self, content: &str) {
        self.history.push(ChatMessage {
            role: Role::Assistant,
            content: content.trim().to_string(),
        });
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Formats the conversation history into a structured token sequence:
    /// <system>\n{system_prompt}\n</system>\n
    /// <user>\n{user_message}\n</user>\n
    /// <assistant>\n{assistant_response}\n</assistant>\n
    /// ...
    /// <assistant>\n
    pub fn format_prompt(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        let system_text = format!("<system>\n{}\n</system>\n", self.system_prompt);
        let system_tokens = tokenizer.encode(&system_text);

        // Encode all history turns
        let mut turns_tokens: Vec<Vec<u32>> = Vec::new();
        for msg in &self.history {
            let turn_text = match msg.role {
                Role::User => format!("<user>\n{}\n</user>\n", msg.content),
                Role::Assistant => format!("<assistant>\n{}\n</assistant>\n", msg.content),
                Role::System => format!("<system>\n{}\n</system>\n", msg.content),
            };
            turns_tokens.push(tokenizer.encode(&turn_text));
        }

        // Assistant generation prompt trigger
        let trigger_tokens = tokenizer.encode("<assistant>\n");

        // Fit turns from most recent into budget
        let fixed_overhead = system_tokens.len() + trigger_tokens.len();
        let available_budget = self.max_context_tokens.saturating_sub(fixed_overhead);

        let mut included_turns: Vec<Vec<u32>> = Vec::new();
        let mut used_tokens = 0;

        for turn in turns_tokens.iter().rev() {
            if used_tokens + turn.len() <= available_budget {
                included_turns.push(turn.clone());
                used_tokens += turn.len();
            } else {
                break;
            }
        }
        included_turns.reverse();

        // Assemble final prompt tokens
        let mut final_tokens = system_tokens;
        for turn in included_turns {
            final_tokens.extend(turn);
        }
        final_tokens.extend(trigger_tokens);
        final_tokens
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize conversation history")?;
        fs::write(path, json).context("Failed to write conversation history file")?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path).context("Failed to read conversation history file")?;
        let mgr: Self = serde_json::from_str(&json).context("Failed to parse conversation history JSON")?;
        Ok(mgr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_formatting() {
        let tok = Tokenizer::new_base(crate::tokenizer::BASE_VOCAB_SIZE as usize);
        let mut mgr = ConversationManager::new("Mneme Sovereign Agent", 256);
        mgr.add_user_message("What is memory?");
        mgr.add_assistant_message("Memory is the retention of information over time.");
        mgr.add_user_message("Can machines have it?");

        let prompt_tokens = mgr.format_prompt(&tok);
        let decoded = tok.decode(&prompt_tokens);

        assert!(decoded.contains("<system>\nMneme Sovereign Agent\n</system>"));
        assert!(decoded.contains("<user>\nWhat is memory?\n</user>"));
        assert!(decoded.contains("<assistant>\nMemory is the retention of information over time.\n</assistant>"));
        assert!(decoded.contains("<user>\nCan machines have it?\n</user>"));
        assert!(decoded.ends_with("<assistant>\n"));
    }

    #[test]
    fn test_conversation_budget_pruning() {
        let tok = Tokenizer::new_base(crate::tokenizer::BASE_VOCAB_SIZE as usize);
        let mut mgr = ConversationManager::new("SysPrompt", 80);
        for i in 1..=10 {
            mgr.add_user_message(&format!("Question number {}", i));
            mgr.add_assistant_message(&format!("Answer number {}", i));
        }

        let prompt_tokens = mgr.format_prompt(&tok);
        assert!(prompt_tokens.len() <= 80);
        let decoded = tok.decode(&prompt_tokens);
        // The most recent turns should be present
        assert!(decoded.contains("Question number 10"));
    }
}
