"""A NumPy reference for ember's forward pass — the algorithm oracle.

This mirrors `src/{model,ops,attention,sample}.rs` exactly: bf16->f32 weight
loading, RMSNorm, HuggingFace rotate-half RoPE, grouped-query attention with a
causal KV cache, SwiGLU, a tied LM head, and greedy decoding. If the Rust engine
and this script produce the same continuation, the port is correct.

It also stands alone as a self-contained, dependency-light description of what
the engine does.

    pip install numpy tokenizers
    python scripts/reference_forward.py --model weights --prompt "The capital of France is"

Verified output (Qwen2.5-0.5B-Instruct, greedy):
    'The capital of France is Paris. It is the largest city in Europe and the third'
"""
import argparse
import json
import struct
import sys
import time

import numpy as np
from tokenizers import Tokenizer

EOS = (151643, 151645)


def load_safetensors(path):
    blob = open(path, "rb").read()
    n = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + n])
    base = 8 + n

    def tensor(name):
        h = header[name]
        a, b = h["data_offsets"]
        buf = blob[base + a: base + b]
        if h["dtype"] == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
            arr = (u << 16).view(np.float32)
        elif h["dtype"] == "F32":
            arr = np.frombuffer(buf, dtype=np.float32)
        elif h["dtype"] == "F16":
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        else:
            raise ValueError(f"unsupported dtype {h['dtype']}")
        return arr.reshape(h["shape"])

    return tensor, header


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="weights")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the Qwen ChatML template")
    args = ap.parse_args()

    c = json.load(open(f"{args.model}/config.json"))
    H, L = c["hidden_size"], c["num_hidden_layers"]
    NH, NKV = c["num_attention_heads"], c["num_key_value_heads"]
    HD = H // NH
    EPS, THETA = c["rms_norm_eps"], c["rope_theta"]
    GROUP, KVD = NH // NKV, NKV * (H // NH)

    print(f"loading {args.model} (H={H} L={L} NH={NH} NKV={NKV} HD={HD}) ...", file=sys.stderr)
    t0 = time.time()
    tensor, _ = load_safetensors(f"{args.model}/model.safetensors")
    embed = tensor("model.embed_tokens.weight")
    fnorm = tensor("model.norm.weight")
    layers = []
    for i in range(L):
        p = f"model.layers.{i}"
        layers.append(dict(
            iln=tensor(f"{p}.input_layernorm.weight"),
            qw=tensor(f"{p}.self_attn.q_proj.weight"), qb=tensor(f"{p}.self_attn.q_proj.bias"),
            kw=tensor(f"{p}.self_attn.k_proj.weight"), kb=tensor(f"{p}.self_attn.k_proj.bias"),
            vw=tensor(f"{p}.self_attn.v_proj.weight"), vb=tensor(f"{p}.self_attn.v_proj.bias"),
            ow=tensor(f"{p}.self_attn.o_proj.weight"),
            pln=tensor(f"{p}.post_attention_layernorm.weight"),
            gw=tensor(f"{p}.mlp.gate_proj.weight"), uw=tensor(f"{p}.mlp.up_proj.weight"),
            dw=tensor(f"{p}.mlp.down_proj.weight"),
        ))
    print(f"loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    half = HD // 2
    inv_freq = 1.0 / (THETA ** (2 * np.arange(half) / HD))

    def rmsnorm(x, w):
        return x / np.sqrt((x * x).mean() + EPS) * w

    def rope(v, pos):  # HF rotate-half convention
        cos, sin = np.cos(pos * inv_freq), np.sin(pos * inv_freq)
        x0, x1 = v[:half].copy(), v[half:].copy()
        out = v.copy()
        out[:half] = x0 * cos - x1 * sin
        out[half:] = x1 * cos + x0 * sin
        return out

    kc = [np.zeros((0, KVD), np.float32) for _ in range(L)]
    vc = [np.zeros((0, KVD), np.float32) for _ in range(L)]
    scale = 1.0 / np.sqrt(HD)

    def forward(tok, pos):
        x = embed[tok].astype(np.float32).copy()
        for li, ly in enumerate(layers):
            xb = rmsnorm(x, ly["iln"])
            q = ly["qw"] @ xb + ly["qb"]
            k = ly["kw"] @ xb + ly["kb"]
            v = ly["vw"] @ xb + ly["vb"]
            for h in range(NH):
                q[h*HD:(h+1)*HD] = rope(q[h*HD:(h+1)*HD], pos)
            for h in range(NKV):
                k[h*HD:(h+1)*HD] = rope(k[h*HD:(h+1)*HD], pos)
            kc[li] = np.vstack([kc[li], k[None]])
            vc[li] = np.vstack([vc[li], v[None]])
            att = np.zeros(NH * HD, np.float32)
            for h in range(NH):
                kh = h // GROUP
                sc = (kc[li][:, kh*HD:(kh+1)*HD] @ q[h*HD:(h+1)*HD]) * scale
                sc = np.exp(sc - sc.max()); sc /= sc.sum()
                att[h*HD:(h+1)*HD] = sc @ vc[li][:, kh*HD:(kh+1)*HD]
            x = x + ly["ow"] @ att
            xb = rmsnorm(x, ly["pln"])
            gate = ly["gw"] @ xb
            act = (gate / (1 + np.exp(-gate))) * (ly["uw"] @ xb)
            x = x + ly["dw"] @ act
        x = rmsnorm(x, fnorm)
        return embed @ x  # tied LM head

    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    prompt = args.prompt
    if args.chat:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    ids = tok.encode(prompt, add_special_tokens=False).ids
    logits = None
    for pos, i in enumerate(ids):
        logits = forward(i, pos)
    out = list(ids)
    for _ in range(args.max_tokens):
        nxt = int(np.argmax(logits))
        if nxt in EOS:
            break
        out.append(nxt)
        logits = forward(nxt, len(out) - 1)

    print(tok.decode(out))


if __name__ == "__main__":
    main()
