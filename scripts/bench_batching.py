"""Throughput of the continuous-batching engine vs serving requests one at a time.

    python scripts/bench_batching.py
"""

from __future__ import annotations

import time

import torch

from microgpt.batching import BatchEngine
from microgpt.model import GPT, GPTConfig


def main() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=96, block_size=256, n_layer=6, n_head=6, n_embd=192, n_kv_head=2)
    model = GPT(cfg).eval()
    g = torch.Generator().manual_seed(0)
    reqs = []
    for _ in range(16):
        length = int(torch.randint(8, 48, (1,), generator=g))
        reqs.append((torch.randint(0, 96, (length,), generator=g).tolist(), 64))
    total = sum(n for _, n in reqs)
    print(f"{len(reqs)} requests, prompts 8-47 tokens, 64 new tokens each ({total} total)\n")
    print("| strategy | wall (s) | tokens/s | speedup | mean batch |")
    print("|:--|--:|--:|--:|--:|")

    model.generate(torch.tensor([reqs[0][0]]), 8, temperature=0)  # warm up
    t0 = time.perf_counter()
    for prompt, n in reqs:
        model.generate(torch.tensor([prompt]), n, temperature=0)
    base = time.perf_counter() - t0
    print(f"| one at a time | {base:.1f} | {total / base:.0f} | 1.0x | 1.0 |")

    for max_batch in (2, 4, 8, 16):
        eng = BatchEngine(model, max_batch=max_batch)
        for prompt, n in reqs:
            eng.submit(prompt, n)
        t0 = time.perf_counter()
        eng.run()
        dt = time.perf_counter() - t0
        print(
            f"| continuous, max_batch={max_batch} | {dt:.1f} | {total / dt:.0f} "
            f"| {base / dt:.1f}x | {eng.mean_batch_size:.1f} |"
        )


if __name__ == "__main__":
    main()
