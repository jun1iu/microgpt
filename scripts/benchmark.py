"""Measure decoding throughput with and without the KV cache.

    python scripts/benchmark.py            # prints a markdown table
"""

from __future__ import annotations

import time

import torch

from microgpt.model import GPT, GPTConfig


def time_generate(model: GPT, prompt: torch.Tensor, n: int, use_cache: bool) -> float:
    t0 = time.perf_counter()
    model.generate(prompt, n, temperature=0, use_cache=use_cache)
    return time.perf_counter() - t0


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = GPTConfig(vocab_size=96, block_size=512, n_layer=6, n_head=6, n_embd=192)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 16))
    print(f"model: {model.num_params() / 1e6:.2f}M params, {torch.get_num_threads()} CPU threads\n")
    for use_cache in (False, True):  # warm up allocator / thread pool before timing
        time_generate(model, prompt, 32, use_cache)
    print("| new tokens | no cache (tok/s) | KV cache (tok/s) | speedup | identical output |")
    print("|-----------:|-----------------:|-----------------:|--------:|:-----------------:|")
    for n in (32, 64, 128, 256, 480):
        slow = time_generate(model, prompt, n, use_cache=False)
        fast = time_generate(model, prompt, n, use_cache=True)
        same = torch.equal(
            model.generate(prompt, n, temperature=0, use_cache=False),
            model.generate(prompt, n, temperature=0, use_cache=True),
        )
        verdict = "yes" if same else "NO"
        print(f"| {n} | {n / slow:.1f} | {n / fast:.1f} | {slow / fast:.1f}x | {verdict} |")


if __name__ == "__main__":
    main()
