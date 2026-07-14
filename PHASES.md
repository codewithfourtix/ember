# Build plan — three phases

`ember` is built in three phases. Each ends with something that works and is worth showing.

## Phase 1 — Correctness (a real LLM, generating text) ✅

A hand-written **f32 forward pass** that loads a real Qwen2.5 model and produces
coherent text. Nothing fast yet — just *right*.

- Load weights from `model.safetensors` (mmap) and the tokenizer from `tokenizer.json`
- Kernels: `matvec`, `rms_norm`, `rope`, grouped-query `attention` + KV cache, `swiglu`, `softmax`
- The full transformer forward pass in `model.rs`
- Greedy + top-p sampling, wired into a real generation loop
- **Done when:** `cargo run --release -- --prompt "The capital of France is"` prints a
  coherent continuation. **✅ Done** — output: *"…is Paris. It is the largest city in
  Europe and the third…"*, cross-checked against `scripts/reference_forward.py`.

## Phase 2 — Performance (fast, and quantized)

Make it quick and small.

- `rayon`-parallel `matvec`; tighten the KV-cache path
- **INT8** then **INT4** row-wise weight quantization with a fused dequant mat-vec
- A benchmark harness reporting tokens/sec and peak memory
- **Done when:** a table shows the speedup and the ~4× memory cut vs the f32 baseline.
  **Done** — INT8 4.0× and grouped-INT4 7.1× memory, both verified coherent, plus a
  `--bench` harness with tokens/sec measured in CI (see the README table). Quantization is
  a memory win; a SIMD dequant to also win throughput is a future optimization.

## Phase 3 — Polish & ship ✅

Make it a thing people can use and be impressed by.

- [x] Streaming token output + ChatML chat mode (`--chat`, `--system`) + nicer CLI
- [x] Benchmark table + demo in the README
- [x] Unit tests for the kernels (`cargo test`) — mat-vec, RMSNorm, RoPE, SwiGLU, softmax,
      sampling, and quantization round-trips, against small hand-checked cases
- [ ] Write-up; optional: a merged perf PR to a major serving framework (vLLM / SGLang / llama.cpp)

Chat mode verified against the NumPy reference — e.g. *"Write a haiku about Rust
programming."* → *"Rust's syntax shines, / Type-safe, concise, and fast, / Programming heaven."*
