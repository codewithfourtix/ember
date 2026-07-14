//! Model hyper-parameters, parsed from a HuggingFace `config.json`.
//!
//! Only the fields the inference path actually needs are read; everything else
//! in the file is ignored. Defaults match the Qwen2.5 family.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// The subset of `config.json` needed to run a Llama-style decoder.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Model (embedding) dimension.
    pub hidden_size: usize,
    /// Number of stacked transformer blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (`< num_attention_heads` ⇒ grouped-query attention).
    pub num_key_value_heads: usize,
    /// Hidden dimension of the SwiGLU feed-forward network.
    pub intermediate_size: usize,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Context length the RoPE tables are built for.
    pub max_position_embeddings: usize,
    /// RoPE base frequency (θ).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Whether the LM head shares weights with the input embedding.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

impl Config {
    /// Load and parse a `config.json` from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config from {}", path.display()))?;
        serde_json::from_str(&text).context("parsing config.json")
    }

    /// Dimension of a single attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// How many query heads share each key/value head (the GQA group size).
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Combined width of the key (or value) projection: `num_key_value_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    /// The Qwen2.5-0.5B-Instruct shape — used by the benchmark to build a
    /// randomly-weighted model without needing the real weights on disk.
    pub fn preset_qwen2_5_0_5b() -> Self {
        Config {
            hidden_size: 896,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            intermediate_size: 4864,
            vocab_size: 151936,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
        }
    }
}
