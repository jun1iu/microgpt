"""Next-token sampling: greedy, temperature, top-k and top-p (nucleus)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits per row; set the rest to -inf."""
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probability mass reaches p (always >= 1 token)."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    # Mass strictly before each token: drop a token if the ones ahead of it already cover p.
    mass_before = probs.cumsum(dim=-1) - probs
    sorted_logits = sorted_logits.masked_fill(mass_before >= p, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """The exact distribution a sampler draws from: temperature, then top-k, then top-p.

    temperature == 0 gives a one-hot at the argmax, so greedy decoding is just the
    degenerate case of sampling (speculative decoding relies on this uniform view).
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        return F.one_hot(logits.argmax(dim=-1), logits.size(-1)).to(logits.dtype)
    logits = logits / temperature
    if top_k is not None:
        logits = filter_top_k(logits, top_k)
    if top_p is not None:
        logits = filter_top_p(logits, top_p)
    return F.softmax(logits, dim=-1)


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one token id per row from logits (B, V). Returns (B, 1).

    temperature == 0 means greedy argmax. Filters are applied after temperature scaling.
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = logits_to_probs(logits, temperature, top_k, top_p)
    return torch.multinomial(probs, num_samples=1, generator=generator)
