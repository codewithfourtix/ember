<div align="center">

# 🔥 ember

**Run a real LLM on your CPU — a from-scratch inference engine in Rust.**

[![Rust](https://img.shields.io/badge/Rust-2021-CE422B?style=flat-square&logo=rust&logoColor=white&labelColor=0d0e11)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square&labelColor=0d0e11)](LICENSE)
[![status](https://img.shields.io/badge/status-scaffolding-f0ad4e?style=flat-square&labelColor=0d0e11)](#roadmap)

</div>

`ember` loads a **Qwen2.5** model and generates text with a hand-written transformer
forward pass — RMSNorm, RoPE, grouped-query attention with a KV cache, and a SwiGLU
MLP — then runs the heavy matrices through custom **INT8/INT4 quantization** so a
0.5–1.5B model fits in a laptop's memory and decodes fast. **No `candle`, `burn`,
`tch`, or `ndarray` in the core:** the transformer math is the project.

> **Status — Phase 1 complete.** The f32 forward pass is implemented and verified end
> to end: run against Qwen2.5-0.5B it generates coherent text. Speed (quantization,
> benchmarks) is Phase 2 — see [`PHASES.md`](PHASES.md).

```console
$ ember --prompt "The capital of France is"
The capital of France is Paris. It is the largest city in Europe and the third
```

<sub>Greedy decode from Qwen2.5-0.5B, cross-checked token-for-token against the NumPy reference in <a href="scripts/reference_forward.py"><code>scripts/reference_forward.py</code></a>.</sub>

## Why build this

Writing an inference engine is the clearest way to understand — and to demonstrate
understanding of — how modern LLMs actually run: attention, the KV cache, the
memory-bandwidth wall of CPU decoding, and quantization. The only dependencies here
are for plumbing (weight loading, tokenization, threading); every kernel is hand-written.

## Design

| Module | Responsibility |
|---|---|
| [`config.rs`](src/config.rs) | Parse the model's `config.json` (Qwen2.5 / Llama-style). |
| [`tensor.rs`](src/tensor.rs) | The hot loop — hand-written, `rayon`-parallel mat-vec. |
| [`ops.rs`](src/ops.rs) | RMSNorm, RoPE, SwiGLU, softmax. |
| [`attention.rs`](src/attention.rs) | Grouped-query attention + the rolling KV cache. |
| [`quant.rs`](src/quant.rs) | Row-wise INT8/INT4 quantization + fused dequant mat-vec. |
| [`sample.rs`](src/sample.rs) | Greedy / temperature / top-p sampling. |
| [`model.rs`](src/model.rs) | safetensors weight loading + the full forward pass. |
| [`main.rs`](src/main.rs) | CLI and the generation loop. |

## Quickstart

```bash
# 1. Rust toolchain (https://rustup.rs)
rustup default stable

# 2. Fetch a small model  (needs: pip install huggingface_hub)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  model.safetensors config.json tokenizer.json --local-dir ./weights

# 3. Build & run
cargo run --release -- --prompt "The capital of France is" --model ./weights
```

## Correctness oracle

Before trusting the generation loop, match a **single forward pass** against
HuggingFace `transformers`:

```bash
pip install torch transformers
python scripts/reference_logits.py --model Qwen/Qwen2.5-0.5B-Instruct
```

`ember`'s next-token logits for the same prompt should agree to ~`1e-3`. Reach parity
here first and every remaining bug is in the loop, not the model.

## Roadmap

See [`PHASES.md`](PHASES.md) for the full plan.

- [x] **Phase 1 — Correctness** — safetensors loading, tokenizer, all kernels (RMSNorm,
      RoPE, GQA attention + KV cache, SwiGLU), the full forward pass, sampling, and the
      generation loop. Verified: generates coherent text from Qwen2.5-0.5B.
- [ ] **Phase 2 — Performance** — `rayon` tuning, INT8/INT4 quantization, tok/s + memory benchmarks
- [ ] **Phase 3 — Polish & ship** — streaming CLI, chat template, benchmark table, tests, write-up

## License

MIT © [Ali Zulfiqar](https://github.com/codewithfourtix)
