//! `ember` — a from-scratch LLM inference engine.
//!
//! Loads a Qwen2.5 model and generates text on the CPU with a hand-written
//! transformer forward pass. This file wires the CLI and tokenizer to the
//! decode loop; the numeric work lives in the sibling modules.

#![allow(dead_code)]

mod attention;
mod chat;
mod config;
mod model;
mod ops;
mod quant;
mod sample;
mod tensor;

use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use tokenizers::Tokenizer;

use model::Model;
use quant::Quant;
use sample::{Rng, Sampler};

/// Token ids that end generation (Qwen2.5: <|endoftext|> and <|im_end|>).
const EOS_IDS: [u32; 2] = [151643, 151645];

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

    /// Weight quantization: `none` (f32), `int8`, or `int4`.
    #[arg(short, long, default_value = "none")]
    quant: String,

    /// Treat the prompt as a chat turn (Qwen ChatML template).
    #[arg(long)]
    chat: bool,

    /// System prompt used in `--chat` mode.
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let scheme = Quant::parse(&args.quant)
        .ok_or_else(|| anyhow!("unknown --quant '{}' (expected none, int8, or int4)", args.quant))?;

    eprintln!("loading model from {} ({}) ...", args.model.display(), args.quant);
    let load_start = Instant::now();
    let model = Model::load(&args.model, scheme)
        .with_context(|| format!("loading model from {}", args.model.display()))?;
    let tokenizer = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow!("loading tokenizer: {e}"))?;
    eprintln!(
        "loaded {} layers, {:.0} MB weights, in {:.1}s",
        model.config.num_hidden_layers,
        model.weight_bytes() as f64 / 1e6,
        load_start.elapsed().as_secs_f32()
    );

    let sampler = if args.temperature <= 0.0 {
        Sampler::Greedy
    } else {
        Sampler::TopP { temperature: args.temperature, top_p: args.top_p }
    };
    let mut rng = Rng::new(seed());

    // In chat mode, wrap the prompt in the ChatML template; otherwise complete it raw.
    let full_prompt = if args.chat {
        chat::chatml(&args.system, &args.prompt)
    } else {
        args.prompt.clone()
    };
    let encoding = tokenizer
        .encode(full_prompt.as_str(), false)
        .map_err(|e| anyhow!("tokenizing prompt: {e}"))?;
    let prompt_ids = encoding.get_ids();
    if prompt_ids.is_empty() {
        return Err(anyhow!("empty prompt"));
    }

    let mut cache = model.new_cache();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    // Prefill: feed every prompt token, keeping the logits after the last one.
    // In completion mode we echo the prompt; in chat mode we stream only the reply.
    if !args.chat {
        print!("{}", args.prompt);
        out.flush().ok();
    }
    let gen_start = Instant::now();
    let mut logits = Vec::new();
    for (pos, &id) in prompt_ids.iter().enumerate() {
        logits = model.forward(id, pos, &mut cache);
        cache.advance();
    }

    // Decode: sample, emit, feed back.
    let mut pos = prompt_ids.len();
    let mut generated = 0usize;
    for _ in 0..args.max_tokens {
        let next = sampler.sample(&logits, &mut rng);
        if EOS_IDS.contains(&next) {
            break;
        }
        let piece = tokenizer
            .decode(&[next], false)
            .map_err(|e| anyhow!("decoding token: {e}"))?;
        print!("{piece}");
        out.flush().ok();

        logits = model.forward(next, pos, &mut cache);
        cache.advance();
        pos += 1;
        generated += 1;
    }
    println!();

    let secs = gen_start.elapsed().as_secs_f32();
    eprintln!(
        "\n{generated} tokens in {secs:.1}s ({:.1} tok/s)",
        generated as f32 / secs.max(1e-6)
    );
    Ok(())
}

/// A time-based seed for the sampler's RNG.
fn seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678)
}
