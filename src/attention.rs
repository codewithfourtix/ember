//! Grouped-query self-attention with a KV cache.
//!
//! At decode time each new token attends over every previous token, so the keys
//! and values for the whole sequence are cached and only the *new* token's
//! Q/K/V are computed each step — the difference between O(n²) and O(n) work per
//! token. Qwen2.5 uses grouped-query attention: several query heads share one
//! key/value head, so the cache stores only `num_key_value_heads` streams.

use crate::config::Config;
use crate::ops::softmax;

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

    /// The key stream for `layer`.
    pub fn keys(&self, layer: usize) -> &[f32] {
        &self.keys[layer]
    }

    /// The value stream for `layer`.
    pub fn values(&self, layer: usize) -> &[f32] {
        &self.values[layer]
    }

    /// Write this step's key/value for `layer` at the current write position.
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
/// `q`/`k`/`v` are this token's projections (post-RoPE for `q`/`k`). This token's
/// key/value are written into the cache, then each query head attends over
/// positions `0..=pos` and the softmax-weighted values are written into `out`.
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
    let hd = config.head_dim();
    let n_heads = config.num_attention_heads;
    let n_kv = config.num_key_value_heads;
    let group = n_heads / n_kv;
    let kv_dim = config.kv_dim();
    let scale = 1.0 / (hd as f32).sqrt();
    let seq = pos + 1;

    // Store this token's key/value, then read the whole (now length-`seq`) cache.
    cache.store(layer, k, v);
    let keys = cache.keys(layer);
    let values = cache.values(layer);

    let mut scores = vec![0.0f32; seq];
    for h in 0..n_heads {
        let kh = h / group; // which kv head this query head shares
        let qh = &q[h * hd..h * hd + hd];

        // scores[t] = (q_h · k_t) / sqrt(head_dim)
        for t in 0..seq {
            let base = t * kv_dim + kh * hd;
            let kt = &keys[base..base + hd];
            let mut s = 0.0f32;
            for i in 0..hd {
                s += qh[i] * kt[i];
            }
            scores[t] = s * scale;
        }
        softmax(&mut scores);

        // out_h = Σ_t scores[t] * v_t
        let oh = &mut out[h * hd..h * hd + hd];
        oh.iter_mut().for_each(|o| *o = 0.0);
        for t in 0..seq {
            let base = t * kv_dim + kh * hd;
            let vt = &values[base..base + hd];
            let w = scores[t];
            for i in 0..hd {
                oh[i] += w * vt[i];
            }
        }
    }
}
