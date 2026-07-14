//! The model: weights loaded from safetensors, and the forward pass that ties
//! every kernel together into a single decode step.

use std::path::Path;

use anyhow::{Context, Result};

use crate::attention::KvCache;
use crate::config::Config;

/// A loaded model: hyper-parameters plus the raw weight tensors.
///
/// Weights are held as `f32` for the initial (correctness-first) build; the
/// quantized path in [`crate::quant`] later replaces the hot matrices.
pub struct Model {
    pub config: Config,
    // Once loading is implemented this holds the per-layer and global tensors:
    // token embedding, per-block attention (q/k/v/o) and MLP (gate/up/down)
    // projections, the two RMSNorm weights per block, the final norm, and the
    // LM head (or a reference to the tied embedding).
}

impl Model {
    /// Load a model from a directory containing `config.json`,
    /// `model.safetensors`, and `tokenizer.json`.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Config::from_file(dir.join("config.json"))
            .with_context(|| format!("loading model from {}", dir.display()))?;
        // Day 1: mmap `model.safetensors`, resolve every tensor by name, and
        // keep the buffers this struct needs for the forward pass.
        Ok(Self { config })
    }

    /// Run one decode step: embed `token` at absolute position `pos`, run every
    /// transformer block (updating `cache`), and return logits over the vocab.
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        // embed → N × (RMSNorm → GQA attention → residual → RMSNorm → SwiGLU →
        // residual) → final RMSNorm → LM head.
        let _ = (token, pos, cache);
        todo!("full transformer forward pass")
    }

    /// A KV cache sized for this model's maximum context length.
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(&self.config, self.config.max_position_embeddings)
    }
}
