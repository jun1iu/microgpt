"""Speculative decoding (Leviathan et al. 2023; Chen et al. 2023).

A cheap *draft* model proposes `gamma` tokens; the expensive *target* model scores all of them
in a single forward pass and keeps the longest prefix it agrees with. Output is distributed
**exactly** as if sampled from the target alone, but the target runs once per round instead of
once per token.

Acceptance rule, for proposal x_i drawn from the draft distribution q_i, with target p_i:
    accept with probability min(1, p_i(x_i) / q_i(x_i));
    on the first rejection, resample from the residual  norm(max(0, p_i - q_i))  and stop;
    if every proposal is accepted, draw one bonus token from the target's next distribution.
Both p and q are the *processed* distributions (after temperature / top-k / top-p), so the
guarantee holds for whatever sampler is configured. Temperature 0 reduces to exact greedy.

Cache bookkeeping: both KV caches are rolled back (truncated) to the accepted prefix after each
round. The draft's cache may lag by one token (the last proposal is never fed), so each round the
draft simply consumes `seq[d_len:]`, which makes catching up automatic.

Single sequence only, and the whole run must fit in `block_size` (no sliding window).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from microgpt.model import GPT, KVCache
from microgpt.sampling import logits_to_probs


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    new_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_call(self) -> float:
        return self.new_tokens / self.target_calls if self.target_calls else 0.0


def _truncate(cache: KVCache, n: int) -> KVCache:
    return [(k[:, :, :n], v[:, :, :n]) for k, v in cache]


@torch.no_grad()
def speculative_generate(
    target: GPT,
    draft: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, SpecStats]:
    """Extend idx (1, T) by max_new_tokens tokens. Returns (tokens, stats)."""
    if idx.dim() != 2 or idx.size(0) != 1:
        raise ValueError("speculative_generate supports a single sequence: idx must be (1, T)")
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("target and draft must share a vocabulary")
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    limit = min(target.cfg.block_size, draft.cfg.block_size)
    if idx.size(1) + max_new_tokens > limit:
        raise ValueError(f"prompt + max_new_tokens exceeds the shorter block_size ({limit})")

    modes = (target.training, draft.training)
    target.eval()
    draft.eval()

    def probs(logits: torch.Tensor) -> torch.Tensor:
        return logits_to_probs(logits, temperature, top_k, top_p)

    seq = idx.clone()
    goal = idx.size(1) + max_new_tokens
    t_cache: Optional[KVCache] = None
    d_cache: Optional[KVCache] = None
    t_len = d_len = 0  # tokens of `seq` already inside each cache
    stats = SpecStats()

    while seq.size(1) < goal:
        n_old = seq.size(1)
        g = min(gamma, goal - n_old - 1)  # always leave room for the guaranteed extra token

        # 1) Draft proposes g tokens autoregressively, remembering its distributions q_i.
        proposals, q_dists = [], []
        if g > 0:
            feed = seq[:, d_len:]
            for _ in range(g):
                logits, _, d_cache = draft(feed, cache=d_cache)
                d_len += feed.size(1)
                stats.draft_calls += 1
                q = probs(logits[:, -1])[0]
                x = torch.multinomial(q, 1, generator=generator).view(1, 1)
                proposals.append(x)
                q_dists.append(q)
                feed = x

        # 2) Target scores the uncached suffix plus every proposal in ONE forward pass.
        feed = seq[:, t_len:]
        if g > 0:
            feed = torch.cat([feed] + proposals, dim=1)
        logits, _, t_cache = target(feed, cache=t_cache)
        t_len += feed.size(1)
        stats.target_calls += 1
        p_dists = probs(logits[0, -(g + 1) :])  # rows 0..g-1 judge x_1..x_g; row g is the bonus

        # 3) Accept / reject left to right.
        n_acc, extra = 0, None
        for i in range(g):
            x = int(proposals[i])
            p, q = p_dists[i], q_dists[i]
            if torch.rand((), generator=generator) < p[x] / q[x]:  # q[x] > 0: x was drawn from q
                n_acc += 1
                continue
            residual = (p - q).clamp_min(0)
            total = residual.sum()
            residual = residual / total if total > 0 else p
            extra = torch.multinomial(residual, 1, generator=generator)
            break
        if extra is None:  # all g accepted (or g == 0): free token from the target
            extra = torch.multinomial(p_dists[g], 1, generator=generator)

        # 4) Commit accepted proposals + the extra token, then roll both caches back.
        new = torch.cat(proposals[:n_acc] + [extra.view(1, 1)], dim=1)
        seq = torch.cat([seq, new], dim=1)
        t_len = n_old + n_acc
        t_cache = _truncate(t_cache, t_len)
        if g > 0 and d_len > n_old + n_acc:
            d_len = n_old + n_acc
            d_cache = _truncate(d_cache, d_len)

        stats.rounds += 1
        stats.proposed += g
        stats.accepted += n_acc
        stats.new_tokens += n_acc + 1

    target.train(modes[0])
    draft.train(modes[1])
    return seq, stats
