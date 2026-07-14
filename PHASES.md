# Build plan — three phases

`ember` is built in three phases. Each ends with something that works and is worth showing.

## Phase 1 — Correctness (a real LLM, generating text)

A hand-written **f32 forward pass** that loads a real Qwen2.5 model and produces
coherent text. Nothing fast yet — just *right*.

- Load weights from `model.safetensors` (mmap) and the tokenizer from `tokenizer.json`
- Kernels: `matvec`, `rms_norm`, `rope`, grouped-query `attention` + KV cache, `swiglu`, `softmax`
- The full transformer forward pass in `model.rs`
- Greedy + top-p sampling, wired into a real generation loop
- **Done when:** `cargo run --release -- --prompt "The capital of France is"` prints a
  coherent continuation.

## Phase 2 — Performance (fast, and quantized)

Make it quick and small.

- `rayon`-parallel `matvec`; tighten the KV-cache path
- **INT8** then **INT4** row-wise weight quantization with a fused dequant mat-vec
- A benchmark harness reporting tokens/sec and peak memory
- **Done when:** a table shows the speedup and the ~4× memory cut vs the f32 baseline.

## Phase 3 — Polish & ship

Make it a thing people can use and be impressed by.

- Streaming token output, chat-template support, nicer CLI
- Benchmark table + a demo in the README
- Unit tests for the kernels (against small hand-checked cases)
- Write-up; optional: a merged perf PR to a major serving framework (vLLM / SGLang / llama.cpp)
