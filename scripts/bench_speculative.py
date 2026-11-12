"""Speculative decoding vs plain target decoding, on the trained Shakespeare models.

    python scripts/bench_speculative.py --target checkpoints/model.pt --draft checkpoints/draft.pt

Reports hardware-independent numbers (acceptance rate, tokens per target forward pass) and
wall-clock speedup, which depends on the draft being much cheaper than the target.
"""

from __future__ import annotations

import argparse
import time

import torch

from microgpt.generate import load_checkpoint
from microgpt.speculative import speculative_generate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="checkpoints/model.pt")
    p.add_argument("--draft", default="checkpoints/draft.pt")
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    target, tok = load_checkpoint(args.target)
    draft, _ = load_checkpoint(args.draft)
    prompts = ["ROMEO:", "First Citizen:\n", "KING RICHARD:\n", "The ", "MENENIUS:\n", "What "]
    print(
        f"target {target.num_params() / 1e6:.2f}M params, draft {draft.num_params() / 1e6:.2f}M, "
        f"temperature {args.temperature}, {args.tokens} new tokens x {args.trials} runs\n"
    )

    def run(fn) -> float:
        fn(0)  # warm up
        t0 = time.perf_counter()
        for i in range(args.trials):
            fn(i)
        return (time.perf_counter() - t0) / args.trials

    def encode(i: int) -> torch.Tensor:
        return torch.tensor([tok.encode(prompts[i % len(prompts)])])

    base = run(
        lambda i: target.generate(
            encode(i),
            args.tokens,
            temperature=args.temperature,
            generator=torch.Generator().manual_seed(i),
        )  # fmt: skip
    )
    print("| method | s / run | speedup | acceptance | tokens / target pass |")
    print("|:--|--:|--:|--:|--:|")
    print(f"| target only | {base:.2f} | 1.00x | - | 1.0 |")

    stats = []
    for gamma in (2, 4, 6, 8):
        stats.clear()

        def spec(i: int, gamma: int = gamma) -> None:
            _, s = speculative_generate(
                target, draft, encode(i), args.tokens, gamma=gamma,
                temperature=args.temperature, generator=torch.Generator().manual_seed(i),
            )  # fmt: skip
            stats.append(s)

        t = run(spec)
        tail = stats[1:]  # drop the warm-up run
        acc = sum(s.accepted for s in tail) / sum(s.proposed for s in tail)
        per_pass = sum(s.new_tokens for s in tail) / sum(s.target_calls for s in tail)
        print(
            f"| speculative, gamma={gamma} | {t:.2f} | {base / t:.2f}x | {acc:.0%} "
            f"| {per_pass:.2f} |"
        )


if __name__ == "__main__":
    main()
