"""KV-cache memory and decode speed for multi-head vs grouped-query vs multi-query attention.

    python scripts/bench_gqa.py
"""

from __future__ import annotations

import time

import torch

from microgpt.model import GPT, GPTConfig, kv_cache_bytes


def main() -> None:
    torch.manual_seed(0)
    n_head, seq, new = 8, 448, 48
    print(f"{n_head} query heads, 6 layers, 256-dim, cache holds {seq + new} tokens\n")
    print("| KV heads | variant | params | KV cache | vs MHA | decode (tok/s) |")
    print("|---:|:--|---:|---:|---:|---:|")
    base = None
    for n_kv in (8, 4, 2, 1):
        cfg = GPTConfig(
            vocab_size=96, block_size=512, n_layer=6, n_head=n_head, n_embd=256, n_kv_head=n_kv
        )
        model = GPT(cfg).eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, seq))
        with torch.no_grad():
            _, _, cache = model(prompt)  # prefill
            tok = prompt[:, -1:]
            t0 = time.perf_counter()
            for _ in range(new):
                logits, _, cache = model(tok, cache=cache)
                tok = logits[:, -1].argmax(-1, keepdim=True)
            rate = new / (time.perf_counter() - t0)
        nbytes = kv_cache_bytes(cache)
        base = base or nbytes
        name = {8: "MHA", 1: "MQA"}.get(n_kv, "GQA")
        print(
            f"| {n_kv} | {name} | {model.num_params() / 1e6:.2f}M | {nbytes / 2**20:.1f} MiB "
            f"| {nbytes / base:.2f}x | {rate:.0f} |"
        )


if __name__ == "__main__":
    main()
