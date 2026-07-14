//! `ember` — a from-scratch LLM inference engine.
//!
//! Loads a Qwen2.5 model and generates text on the CPU. The heavy lifting lives
//! in the sibling modules; this file wires the CLI to the generation loop.
//!
//! The numeric kernels are still `todo!()`, so running the binary today parses
//! args and loads the config, then panics at the first kernel — that panic is
//! the map of what to implement next (see the roadmap in the README).

// Kernels are wired but not yet called from every module during scaffolding.
#![allow(dead_code)]

mod attention;
mod config;
mod model;
mod ops;
mod quant;
mod sample;
mod tensor;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use model::Model;
use sample::Sampler;

/// Run a local LLM on your CPU.
#[derive(Parser, Debug)]
#[command(name = "ember", version, about)]
struct Args {
    /// Prompt to complete.
    #[arg(short, long)]
    prompt: String,

    /// Directory with `config.json`, `model.safetensors`, and `tokenizer.json`.
    #[arg(short, long, default_value = "weights")]
    model: PathBuf,

    /// Maximum number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,

    /// Sampling temperature (0 ⇒ greedy).
    #[arg(short, long, default_value_t = 0.0)]
    temperature: f32,

    /// Nucleus (top-p) cutoff, used when `temperature > 0`.
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model = Model::load(&args.model)
        .with_context(|| format!("loading model from {}", args.model.display()))?;

    let sampler = if args.temperature <= 0.0 {
        Sampler::Greedy
    } else {
        Sampler::TopP {
            temperature: args.temperature,
            top_p: args.top_p,
        }
    };

    // Day 1: load `tokenizer.json`, encode `args.prompt`, prefill the cache with
    // the prompt tokens, then continue the loop below from the last one.
    let mut cache = model.new_cache();
    let mut pos = 0usize;
    let mut token: u32 = 0; // placeholder until the tokenizer is wired

    for _ in 0..args.max_tokens {
        let logits = model.forward(token, pos, &mut cache);
        cache.advance();
        token = sampler.sample(&logits);
        pos += 1;
        // Day 3: decode `token`, stream it to stdout, and stop on the EOS id.
    }

    println!("(generation loop wired — implement the kernels to bring it to life)");
    Ok(())
}
