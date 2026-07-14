#!/usr/bin/env python3
"""Print the next-token logits for a fixed prompt, using HuggingFace transformers.

This is ember's correctness oracle. Run it, then compare against ember's own
logits for the same prompt and token ids — they should agree to ~1e-3. Reach
parity here before trusting the generation loop: after that, every bug is in the
loop, not the model.

    pip install torch transformers
    python scripts/reference_logits.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1]  # logits for the *next* token

    print(f"prompt   : {args.prompt!r}")
    print(f"input ids: {ids[0].tolist()}")
    print(f"logits[:5]: {[round(v, 4) for v in logits[:5].tolist()]}")
    print(f"top-{args.topk} next tokens:")
    top = torch.topk(logits, args.topk)
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {idx:>6}  {score:+8.4f}  {tok.decode([idx])!r}")


if __name__ == "__main__":
    main()
