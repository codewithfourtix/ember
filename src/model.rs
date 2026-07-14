//! The model: weights loaded from safetensors, and the forward pass that ties
//! every kernel together into a single decode step.
//!
//! Every large matrix is a [`Linear`], so the same forward pass runs whether the
//! weights are dense `f32` or INT8/INT4 quantized — the choice is made at load.

use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::attention::{attention, KvCache};
use crate::config::Config;
use crate::ops::{rms_norm, rope, swiglu};
use crate::quant::{Linear, Quant};
use crate::tensor::{add_assign, add_bias, matvec};

/// Cap the KV cache so a huge `max_position_embeddings` doesn't allocate GBs.
const MAX_CONTEXT: usize = 4096;

/// The weights of a single transformer block.
struct Layer {
    input_ln: Vec<f32>,
    q: Linear,
    q_b: Vec<f32>,
    k: Linear,
    k_b: Vec<f32>,
    v: Linear,
    v_b: Vec<f32>,
    o: Linear,
    post_ln: Vec<f32>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// A loaded model: hyper-parameters plus every weight tensor.
pub struct Model {
    pub config: Config,
    /// Token embedding, doubling as the (tied) LM head.
    embed: Linear,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
}

impl Model {
    /// Load a model from a directory containing `config.json` and
    /// `model.safetensors`, quantizing the matrices per `scheme`.
    pub fn load(dir: impl AsRef<Path>, scheme: Quant) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Config::from_file(dir.join("config.json"))
            .with_context(|| format!("loading model from {}", dir.display()))?;

        let path = dir.join("model.safetensors");
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the file is read-only for the lifetime of this load.
        let mmap = unsafe { Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap).context("parsing safetensors")?;

        let f32v = |name: &str| -> Result<Vec<f32>> { load_f32(&st, name) };
        let lin = |name: &str, rows: usize, cols: usize| -> Result<Linear> {
            Ok(Linear::build(load_f32(&st, name)?, rows, cols, scheme))
        };

        let h = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim();
        let kv_dim = config.kv_dim();
        let inter = config.intermediate_size;

        let embed = lin("model.embed_tokens.weight", config.vocab_size, h)?;
        let final_norm = f32v("model.norm.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_ln: f32v(&format!("{p}.input_layernorm.weight"))?,
                q: lin(&format!("{p}.self_attn.q_proj.weight"), q_dim, h)?,
                q_b: f32v(&format!("{p}.self_attn.q_proj.bias"))?,
                k: lin(&format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?,
                k_b: f32v(&format!("{p}.self_attn.k_proj.bias"))?,
                v: lin(&format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?,
                v_b: f32v(&format!("{p}.self_attn.v_proj.bias"))?,
                o: lin(&format!("{p}.self_attn.o_proj.weight"), h, q_dim)?,
                post_ln: f32v(&format!("{p}.post_attention_layernorm.weight"))?,
                gate: lin(&format!("{p}.mlp.gate_proj.weight"), inter, h)?,
                up: lin(&format!("{p}.mlp.up_proj.weight"), inter, h)?,
                down: lin(&format!("{p}.mlp.down_proj.weight"), h, inter)?,
            });
        }

        Ok(Self { config, embed, layers, final_norm })
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
        let mut x = self.embed.row(token as usize);

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

            layer.q.matvec(&xb, &mut q);
            add_bias(&mut q, &layer.q_b);
            layer.k.matvec(&xb, &mut k);
            add_bias(&mut k, &layer.k_b);
            layer.v.matvec(&xb, &mut v);
            add_bias(&mut v, &layer.v_b);

            for head in 0..n_heads {
                rope(&mut q[head * hd..head * hd + hd], pos, hd, theta);
            }
            for head in 0..n_kv {
                rope(&mut k[head * hd..head * hd + hd], pos, hd, theta);
            }

            attention(&q, &k, &v, cache, li, pos, cfg, &mut att);

            layer.o.matvec(&att, &mut proj);
            add_assign(&mut x, &proj);

            // --- Feed-forward block ---
            rms_norm(&x, &layer.post_ln, &mut xb, eps);
            layer.gate.matvec(&xb, &mut gate);
            layer.up.matvec(&xb, &mut up);
            swiglu(&gate, &up, &mut act);
            layer.down.matvec(&act, &mut proj);
            add_assign(&mut x, &proj);
        }

        rms_norm(&x, &self.final_norm, &mut xb, eps);

        // Tied LM head: the embedding matrix, reused as the output projection.
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.embed.matvec(&xb, &mut logits);
        logits
    }

    /// A KV cache sized for this model's (capped) context length.
    pub fn new_cache(&self) -> KvCache {
        let cap = self.config.max_position_embeddings.min(MAX_CONTEXT);
        KvCache::new(&self.config, cap)
    }

    /// Approximate resident weight size in bytes (for the benchmark report).
    pub fn weight_bytes(&self) -> usize {
        let mut total = self.embed.bytes() + self.final_norm.len() * 4;
        for l in &self.layers {
            total += l.q.bytes() + l.k.bytes() + l.v.bytes() + l.o.bytes();
            total += l.gate.bytes() + l.up.bytes() + l.down.bytes();
            total += (l.input_ln.len() + l.post_ln.len() + l.q_b.len() + l.k_b.len() + l.v_b.len()) * 4;
        }
        total
    }
}

/// Load a named tensor and convert it to `f32`, whatever its stored dtype.
fn load_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).map_err(|e| anyhow!("tensor {name}: {e}"))?;
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
