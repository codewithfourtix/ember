//! The model: weights loaded from safetensors, and the forward pass that ties
//! every kernel together into a single decode step.

use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::attention::{attention, KvCache};
use crate::config::Config;
use crate::ops::{rms_norm, rope, swiglu};
use crate::tensor::{add_assign, add_bias, matvec};

/// Cap the KV cache so a huge `max_position_embeddings` doesn't allocate GBs.
const MAX_CONTEXT: usize = 4096;

/// The weights of a single transformer block.
struct Layer {
    input_ln: Vec<f32>,
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    o_w: Vec<f32>,
    post_ln: Vec<f32>,
    gate_w: Vec<f32>,
    up_w: Vec<f32>,
    down_w: Vec<f32>,
}

/// A loaded model: hyper-parameters plus every weight tensor as `f32`.
pub struct Model {
    pub config: Config,
    embed: Vec<f32>,        // [vocab × hidden]
    layers: Vec<Layer>,
    final_norm: Vec<f32>,   // [hidden]
    lm_head: Vec<f32>,      // [vocab × hidden], or empty when tied to `embed`
}

impl Model {
    /// Load a model from a directory containing `config.json` and
    /// `model.safetensors`.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Config::from_file(dir.join("config.json"))
            .with_context(|| format!("loading model from {}", dir.display()))?;

        let path = dir.join("model.safetensors");
        let file = File::open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the file is read-only for the lifetime of this load.
        let mmap = unsafe { Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap).context("parsing safetensors")?;

        let get = |name: &str| -> Result<Vec<f32>> { load_f32(&st, name) };

        let embed = get("model.embed_tokens.weight")?;
        let final_norm = get("model.norm.weight")?;
        // Small Qwen models tie the LM head to the input embedding.
        let lm_head = if config.tie_word_embeddings || st.tensor("lm_head.weight").is_err() {
            Vec::new()
        } else {
            get("lm_head.weight")?
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_ln: get(&format!("{p}.input_layernorm.weight"))?,
                q_w: get(&format!("{p}.self_attn.q_proj.weight"))?,
                q_b: get(&format!("{p}.self_attn.q_proj.bias"))?,
                k_w: get(&format!("{p}.self_attn.k_proj.weight"))?,
                k_b: get(&format!("{p}.self_attn.k_proj.bias"))?,
                v_w: get(&format!("{p}.self_attn.v_proj.weight"))?,
                v_b: get(&format!("{p}.self_attn.v_proj.bias"))?,
                o_w: get(&format!("{p}.self_attn.o_proj.weight"))?,
                post_ln: get(&format!("{p}.post_attention_layernorm.weight"))?,
                gate_w: get(&format!("{p}.mlp.gate_proj.weight"))?,
                up_w: get(&format!("{p}.mlp.up_proj.weight"))?,
                down_w: get(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }

        Ok(Self { config, embed, layers, final_norm, lm_head })
    }

    /// Run one decode step: embed `token` at absolute position `pos`, run every
    /// transformer block (updating `cache`), and return logits over the vocab.
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let hd = cfg.head_dim();
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let q_dim = n_heads * hd;
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;
        let theta = cfg.rope_theta;

        // Residual stream, seeded with the token embedding.
        let mut x = self.embed[token as usize * h..(token as usize + 1) * h].to_vec();

        let mut xb = vec![0.0f32; h];
        let mut q = vec![0.0f32; q_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut att = vec![0.0f32; q_dim];
        let mut proj = vec![0.0f32; h];
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        let mut act = vec![0.0f32; inter];

        for (li, layer) in self.layers.iter().enumerate() {
            // --- Attention block ---
            rms_norm(&x, &layer.input_ln, &mut xb, eps);

            matvec(&layer.q_w, &xb, &mut q, h, q_dim);
            add_bias(&mut q, &layer.q_b);
            matvec(&layer.k_w, &xb, &mut k, h, kv_dim);
            add_bias(&mut k, &layer.k_b);
            matvec(&layer.v_w, &xb, &mut v, h, kv_dim);
            add_bias(&mut v, &layer.v_b);

            for head in 0..n_heads {
                rope(&mut q[head * hd..head * hd + hd], pos, hd, theta);
            }
            for head in 0..n_kv {
                rope(&mut k[head * hd..head * hd + hd], pos, hd, theta);
            }

            attention(&q, &k, &v, cache, li, pos, cfg, &mut att);

            matvec(&layer.o_w, &att, &mut proj, q_dim, h);
            add_assign(&mut x, &proj);

            // --- Feed-forward block ---
            rms_norm(&x, &layer.post_ln, &mut xb, eps);
            matvec(&layer.gate_w, &xb, &mut gate, h, inter);
            matvec(&layer.up_w, &xb, &mut up, h, inter);
            swiglu(&gate, &up, &mut act);
            matvec(&layer.down_w, &act, &mut proj, inter, h);
            add_assign(&mut x, &proj);
        }

        rms_norm(&x, &self.final_norm, &mut xb, eps);

        let lm = if self.lm_head.is_empty() { &self.embed } else { &self.lm_head };
        let mut logits = vec![0.0f32; cfg.vocab_size];
        matvec(lm, &xb, &mut logits, h, cfg.vocab_size);
        logits
    }

    /// A KV cache sized for this model's (capped) context length.
    pub fn new_cache(&self) -> KvCache {
        let cap = self.config.max_position_embeddings.min(MAX_CONTEXT);
        KvCache::new(&self.config, cap)
    }
}

/// Load a named tensor and convert it to `f32`, whatever its stored dtype.
fn load_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st
        .tensor(name)
        .map_err(|e| anyhow!("tensor {name}: {e}"))?;
    let bytes = t.data();
    let out = match t.dtype() {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        other => bail!("tensor {name}: unsupported dtype {other:?}"),
    };
    Ok(out)
}
