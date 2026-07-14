"""Does INT8/INT4 row-wise quantization keep Qwen2.5-0.5B coherent? Measure it.

Quantizes every matmul weight row-wise (symmetric), dequantizes to simulate the
fused inference path, runs greedy generation, and reports output + memory + error.
This is the evidence behind ember's Phase 2 quantization.
"""
import json, struct, sys, time
import numpy as np
from tokenizers import Tokenizer

W = r"C:\Users\aliaz\ember\weights"
blob = open(f"{W}/model.safetensors", "rb").read()
n = struct.unpack("<Q", blob[:8])[0]
hdr = json.loads(blob[8:8 + n]); base = 8 + n
def tens(name):
    h = hdr[name]; a, b = h["data_offsets"]; buf = blob[base+a:base+b]
    u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32).reshape(h["shape"])  # all BF16 here

c = json.load(open(f"{W}/config.json"))
H, L, NH, NKV = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"]
HD, EPS, THETA = H // NH, c["rms_norm_eps"], c["rope_theta"]
GROUP, KVD = NH // NKV, NKV * (H // NH)

def quant_dequant(w, bits, group=0):
    """Symmetric quantize+dequantize. group=0 → per row; else per group of columns."""
    qmax = (1 << (bits - 1)) - 1                     # 127 (int8) or 7 (int4)
    out, cols = w.shape
    g = cols if group in (0, None) else group
    wr = w.reshape(out, cols // g, g)
    scale = np.abs(wr).max(axis=2, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.round(wr / scale), -qmax, qmax)
    return (q * scale).reshape(out, cols).astype(np.float32)

def load(spec=None):
    # spec: None (f32) | (bits, group)
    def maybe(w):
        if spec is None:
            return w.astype(np.float32)
        bits, group = spec
        return quant_dequant(w, bits, group)
    embed = maybe(tens("model.embed_tokens.weight"))
    fnorm = tens("model.norm.weight")
    Ly = []
    for i in range(L):
        p = f"model.layers.{i}"
        Ly.append(dict(
            iln=tens(f"{p}.input_layernorm.weight"),
            qw=maybe(tens(f"{p}.self_attn.q_proj.weight")), qb=tens(f"{p}.self_attn.q_proj.bias"),
            kw=maybe(tens(f"{p}.self_attn.k_proj.weight")), kb=tens(f"{p}.self_attn.k_proj.bias"),
            vw=maybe(tens(f"{p}.self_attn.v_proj.weight")), vb=tens(f"{p}.self_attn.v_proj.bias"),
            ow=maybe(tens(f"{p}.self_attn.o_proj.weight")),
            pln=tens(f"{p}.post_attention_layernorm.weight"),
            gw=maybe(tens(f"{p}.mlp.gate_proj.weight")), uw=maybe(tens(f"{p}.mlp.up_proj.weight")),
            dw=maybe(tens(f"{p}.mlp.down_proj.weight")),
        ))
    return embed, fnorm, Ly

half = HD // 2
inv_freq = 1.0 / (THETA ** (2 * np.arange(half) / HD))
def rmsnorm(x, w): return x / np.sqrt((x * x).mean() + EPS) * w
def rope(v, pos):
    cos, sin = np.cos(pos * inv_freq), np.sin(pos * inv_freq)
    x0, x1 = v[:half].copy(), v[half:].copy(); out = v.copy()
    out[:half] = x0 * cos - x1 * sin; out[half:] = x1 * cos + x0 * sin
    return out

def generate(embed, fnorm, Ly, ids, steps=14):
    kc = [np.zeros((0, KVD), np.float32) for _ in range(L)]
    vc = [np.zeros((0, KVD), np.float32) for _ in range(L)]
    scale = 1.0 / np.sqrt(HD)
    def forward(tok, pos):
        x = embed[tok].astype(np.float32).copy()
        for li, ly in enumerate(Ly):
            xb = rmsnorm(x, ly["iln"])
            q = ly["qw"] @ xb + ly["qb"]; k = ly["kw"] @ xb + ly["kb"]; v = ly["vw"] @ xb + ly["vb"]
            for h in range(NH): q[h*HD:(h+1)*HD] = rope(q[h*HD:(h+1)*HD], pos)
            for h in range(NKV): k[h*HD:(h+1)*HD] = rope(k[h*HD:(h+1)*HD], pos)
            kc[li] = np.vstack([kc[li], k[None]]); vc[li] = np.vstack([vc[li], v[None]])
            att = np.zeros(NH*HD, np.float32)
            for h in range(NH):
                kh = h // GROUP
                sc = (kc[li][:, kh*HD:(kh+1)*HD] @ q[h*HD:(h+1)*HD]) * scale
                sc = np.exp(sc - sc.max()); sc /= sc.sum()
                att[h*HD:(h+1)*HD] = sc @ vc[li][:, kh*HD:(kh+1)*HD]
            x = x + ly["ow"] @ att
            xb = rmsnorm(x, ly["pln"])
            g = ly["gw"] @ xb
            x = x + ly["dw"] @ ((g / (1 + np.exp(-g))) * (ly["uw"] @ xb))
        return embed @ rmsnorm(x, fnorm)
    logits = None
    for pos, i in enumerate(ids): logits = forward(i, pos)
    out = list(ids)
    for _ in range(steps):
        nxt = int(np.argmax(logits))
        if nxt in (151643, 151645): break
        out.append(nxt); logits = forward(nxt, len(out) - 1)
    return out

# --- memory accounting (matmul weights only; norms/biases are tiny) ---
matmul_params = H*151936  # embed / lm head (tied, counted once)
per_layer = H*H + 2*(KVD*H) + H*H + 3*(c["intermediate_size"]*H)
matmul_params += L*per_layer
def mem_mb(spec):
    if spec is None: return matmul_params*4/1e6
    bits, group = spec
    g = 128 if group in (0, None) else group      # scale overhead: one f32 per group
    scale_bytes = (matmul_params / (g if group else 896)) * 4
    return (matmul_params*bits/8 + scale_bytes)/1e6

tok = Tokenizer.from_file(f"{W}/tokenizer.json")
ids = tok.encode("The capital of France is", add_special_tokens=False).ids

print("   scheme | mem (MB) | vs f32 | output", file=sys.stderr)
for label, spec in [("f32", None), ("int8", (8, 0)), ("int4 per-row", (4, 0)),
                    ("int4 g=128", (4, 128)), ("int4 g=64", (4, 64))]:
    embed, fnorm, Ly = load(spec)
    text = tok.decode(generate(embed, fnorm, Ly, ids))
    ratio = mem_mb(None) / mem_mb(spec)
    print(f"{label:>12} | {mem_mb(spec):7.0f}  | {ratio:5.1f}x | {text!r}")
