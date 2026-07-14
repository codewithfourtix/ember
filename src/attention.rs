//! Grouped-query self-attention with a KV cache.
//!
//! At decode time each new token attends over every previous token, so the keys
//! and values for the whole sequence are cached and only the *new* token's
//! Q/K/V are computed each step — the difference between O(n²) and O(n) work per
//! token. Qwen2.5 uses grouped-query attention: several query heads share one
//! key/value head, so the cache stores only `num_key_value_heads` streams.

use crate::config::Config;

/// Per-layer rolling cache of past keys and values.
///
/// Each layer's buffer is laid out as `[pos * kv_dim + i]`, where
/// `kv_dim = num_key_value_heads * head_dim`.
pub struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    kv_dim: usize,
    max_seq: usize,
    len: usize,
}

impl KvCache {
    /// Allocate a cache sized for `config` and a `max_seq`-token context.
    pub fn new(config: &Config, max_seq: usize) -> Self {
        let kv_dim = config.kv_dim();
        let layers = config.num_hidden_layers;
        Self {
            keys: (0..layers).map(|_| vec![0.0; max_seq * kv_dim]).collect(),
            values: (0..layers).map(|_| vec![0.0; max_seq * kv_dim]).collect(),
            kv_dim,
            max_seq,
            len: 0,
        }
    }

    /// Number of tokens currently cached.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache holds no tokens yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The key stream for `layer` (`len * kv_dim` valid entries).
    pub fn keys(&self, layer: usize) -> &[f32] {
        &self.keys[layer]
    }

    /// The value stream for `layer`.
    pub fn values(&self, layer: usize) -> &[f32] {
        &self.values[layer]
    }

    /// Write this step's key/value for `layer` at the current position.
    pub fn store(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);
        debug_assert_eq!(value.len(), self.kv_dim);
        let off = self.len * self.kv_dim;
        self.keys[layer][off..off + self.kv_dim].copy_from_slice(key);
        self.values[layer][off..off + self.kv_dim].copy_from_slice(value);
    }

    /// Advance the write cursor by one token, after every layer has stored its
    /// key/value for this step.
    pub fn advance(&mut self) {
        debug_assert!(self.len < self.max_seq, "KV cache overflow");
        self.len += 1;
    }

    /// Reset for a fresh sequence without reallocating.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

/// Compute self-attention for one token at `pos` in a single layer.
///
/// `q`/`k`/`v` are this token's projections; `k`/`v` are stored into `cache`,
/// then the token attends over positions `0..=pos` and the weighted values are
/// written into `out`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cache: &mut KvCache,
    layer: usize,
    pos: usize,
    config: &Config,
    out: &mut [f32],
) {
    // 1. cache.store(layer, k, v)
    // 2. for each query head h (mapped to kv head h / group_size):
    //      score[t] = (q_h · k_t) / sqrt(head_dim)  for t in 0..=pos
    //      softmax(score); out_h = Σ_t score[t] * v_t
    let _ = (q, k, v, cache, layer, pos, config, out);
    todo!("grouped-query scaled-dot-product attention over the KV cache")
}
