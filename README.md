<div align="center">

# 🔥 ember

**Run a real LLM on your CPU — a from-scratch inference engine in Rust.**

[![CI](https://github.com/codewithfourtix/ember/actions/workflows/ci.yml/badge.svg)](https://github.com/codewithfourtix/ember/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-2021-CE422B?style=flat-square&logo=rust&logoColor=white&labelColor=0d0e11)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square&labelColor=0d0e11)](LICENSE)

</div>

`ember` loads a **Qwen2.5** model and generates text with a hand-written transformer
forward pass — RMSNorm, RoPE, grouped-query attention with a KV cache, and a SwiGLU
MLP — then runs the heavy matrices through custom **INT8/INT4 quantization** so a
0.5–1.5B model fits in a laptop's memory and decodes fast. **No `candle`, `burn`,
`tch`, or `ndarray` in the core:** the transformer math is the project.

> **Status.** All three phases implemented: a verified f32 forward pass, INT8/INT4
> quantization (4× / 7× smaller, still coherent), and a ChatML chat mode. Every kernel
> is cross-checked against a NumPy reference — see [`PHASES.md`](PHASES.md).

```console
$ ember --prompt "The capital of France is"
The capital of France is Paris. It is the largest city in Europe and the third

$ ember --chat --prompt "Write a haiku about Rust programming."
Rust's syntax shines,
Type-safe, concise, and fast,
Programming heaven.
```

<sub>Qwen2.5-0.5B, cross-checked token-for-token against the NumPy reference in <a href="scripts/reference_forward.py"><code>scripts/reference_forward.py</code></a>.</sub>

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

## Quantization (Phase 2)

Row/group-wise symmetric weight quantization, dequantized on the fly inside the
mat-vec. Real numbers on Qwen2.5-0.5B (greedy, prompt *"The capital of France is"*):

| scheme | weights | vs f32 | greedy continuation |
|---|---|---|---|
| `f32` | 1976 MB | 1.0× | …Paris. It is the largest city in Europe and the third largest city |
| **`int8`** (per-row) | **496 MB** | **4.0×** | …Paris. It is the largest city in Europe and the third largest in |
| `int4` per-row | 249 MB | 7.9× | ✗ collapses (*"A. The Eiffel…"*) |
| **`int4`** (group-64) | **278 MB** | **7.1×** | …Paris. It is the largest city in Europe. It is also the |

INT8 is near-lossless; INT4 needs **group-wise** scales — per-row is too coarse and
the output falls apart. Reproduce the table with
[`scripts/quantize_check.py`](scripts/quantize_check.py).

### Metrics — measured on the real Qwen2.5-0.5B

**Quality & memory** — perplexity over a fixed passage (`--perplexity`, lower is better):

| scheme | weights | vs f32 | perplexity ↓ |
|---|---|---|---|
| `none` (f32) | 1976 MB | 1.0× | 6.49 |
| **`int8`** | **496 MB** | **4.0×** | **6.38** — quality-lossless |
| **`int4`** (group-64) | **278 MB** | **7.1×** | 7.75 — +19% |

INT8 quantization is **free** on quality (perplexity is within noise of f32); INT4
trades ~19% perplexity for a 7× memory cut.

**Latency** — greedy, single machine (`--prompt … -n 20`):

| scheme | prefill | decode |
|---|---|---|
| f32 | 8.3 tok/s | 8.6 tok/s · 116 ms/tok |
| int8 | 6.6 tok/s | 6.6 tok/s · 152 ms/tok |

Quantization is a **memory** win; the current scalar dequant is slightly slower per
token, so a SIMD / bandwidth-optimized dequant is the natural next optimization.

```bash
ember --perplexity --quant int8                 # quality metric
ember --bench --quant int8                       # throughput on random weights
ember --prompt "The capital of France is" --quant int8
```

## Roadmap

See [`PHASES.md`](PHASES.md) for the full plan.

- [x] **Phase 1 — Correctness** — safetensors loading, tokenizer, all kernels (RMSNorm,
      RoPE, GQA attention + KV cache, SwiGLU), the full forward pass, sampling, and the
      generation loop. Verified: generates coherent text from Qwen2.5-0.5B.
- [x] **Phase 2 — Performance** — INT8/INT4 quantization (4.0× / 7.1× memory, coherent),
      `rayon`-parallel mat-vec, and a `--bench` throughput harness (numbers above).
- [x] **Phase 3 — Polish & ship** — streaming output, ChatML chat mode (`--chat`), the
      quantization benchmark table above, and `cargo test` kernel unit tests. Tokens/sec
      numbers land once the binary is built on a host without the local toolchain block.

## License

MIT © [Ali Zulfiqar](https://github.com/codewithfourtix)
