"""Continuous batching: serve many variable-length requests from one shared KV cache.

How it works
------------
* Every active sequence is one row of a single batched cache (B, n_kv, S, hd).
* Rows have different lengths, so shorter rows are **left-padded** to the common width S and a
  boolean `mask` (B, S) marks which columns are real. Padding is never attended to.
* RoPE positions are passed explicitly per row (`positions`), so padding does not shift a
  sequence's positions and cached keys keep the rotation they were stored with.
* New requests are prefilled on their own and then spliced into the batch (padding either side
  as needed) while others are mid-decode. Finished requests are dropped from the batch and any
  columns that became padding for *every* remaining row are trimmed away.

Because positions and masking are exact, greedy output for a request is identical to running it
alone with `GPT.generate` (this is what the tests check). With sampling, a shared RNG means
results depend on batch composition, as in any real batched server.

Limitation: a sequence ends when it reaches `block_size` tokens (no sliding window here).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import torch

from microgpt.model import GPT, KVCache
from microgpt.sampling import sample_next


@dataclass
class _Active:
    rid: int
    max_new: int
    eos_id: Optional[int]
    out: List[int]  # generated tokens so far
    cache_len: int  # tokens already in this row's cache == position of the pending token
    pending: int  # last sampled token, not yet fed through the model


def _pad_left(t: torch.Tensor, n: int) -> torch.Tensor:
    """Prepend n zero columns along the sequence axis (dim 2) of a (B, H, S, hd) tensor."""
    if n == 0:
        return t
    pad = torch.zeros(t.size(0), t.size(1), n, t.size(3), dtype=t.dtype, device=t.device)
    return torch.cat((pad, t), dim=2)


def _pad_mask_left(m: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return m
    return torch.cat((torch.zeros(m.size(0), n, dtype=torch.bool, device=m.device), m), dim=1)


class BatchEngine:
    def __init__(
        self,
        model: GPT,
        max_batch: int = 8,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.model = model.eval()
        self.max_batch = max_batch
        self.sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p, generator=generator)
        self.waiting: Deque[Tuple[int, List[int], int, Optional[int]]] = deque()
        self.active: List[_Active] = []
        self.cache: Optional[KVCache] = None
        self.mask: Optional[torch.Tensor] = None  # (B, S) bool, True = real token
        self.results: Dict[int, List[int]] = {}
        self._next_id = 0
        self.stats = {"decode_steps": 0, "generated": 0, "batch_rows": 0, "prefills": 0}

    # ------------------------------------------------------------------ public API
    def submit(self, prompt: List[int], max_new_tokens: int, eos_id: Optional[int] = None) -> int:
        block = self.model.cfg.block_size
        if not 1 <= len(prompt) <= block:
            raise ValueError(f"prompt length must be in [1, {block}], got {len(prompt)}")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        rid = self._next_id
        self._next_id += 1
        self.waiting.append((rid, list(prompt), max_new_tokens, eos_id))
        return rid

    def step(self) -> bool:
        """Admit waiting requests, then advance every active row by one token.

        Returns True while there is still work left."""
        self._admit()
        if self.active:
            self._decode_step()
        return bool(self.waiting or self.active)

    def run(self) -> Dict[int, List[int]]:
        while self.step():
            pass
        return self.results

    @property
    def mean_batch_size(self) -> float:
        steps = self.stats["decode_steps"]
        return self.stats["batch_rows"] / steps if steps else 0.0

    # ------------------------------------------------------------------ internals
    def _sample(self, logits: torch.Tensor) -> List[int]:
        return sample_next(logits, **self.sampling).flatten().tolist()

    def _finished(self, a: _Active) -> bool:
        hit_eos = a.eos_id is not None and a.out[-1] == a.eos_id
        # The pending token needs a free position (< block_size) to be fed next.
        return len(a.out) >= a.max_new or hit_eos or a.cache_len >= self.model.cfg.block_size

    def _admit(self) -> None:
        while self.waiting and len(self.active) < self.max_batch:
            rid, prompt, max_new, eos_id = self.waiting.popleft()
            with torch.no_grad():
                logits, _, cache = self.model(torch.tensor([prompt]))
            first = self._sample(logits[:, -1])[0]
            self.stats["prefills"] += 1
            self.stats["generated"] += 1
            row = _Active(rid, max_new, eos_id, [first], len(prompt), first)
            if self._finished(row):
                self.results[rid] = row.out
            else:
                self._splice(cache, row)

    def _splice(self, cache: KVCache, row: _Active) -> None:
        """Add a freshly prefilled single-row cache to the batch, aligning widths."""
        length = row.cache_len
        if self.cache is None:
            self.cache = cache
            self.mask = torch.ones(1, length, dtype=torch.bool)
        else:
            assert self.mask is not None
            width = self.mask.size(1)
            n = max(width, length)
            self.cache = [
                (
                    torch.cat((_pad_left(k0, n - width), _pad_left(k1, n - length)), dim=0),
                    torch.cat((_pad_left(v0, n - width), _pad_left(v1, n - length)), dim=0),
                )
                for (k0, v0), (k1, v1) in zip(self.cache, cache)
            ]
            self.mask = torch.cat(
                (
                    _pad_mask_left(self.mask, n - width),
                    _pad_mask_left(torch.ones(1, length, dtype=torch.bool), n - length),
                ),
                dim=0,
            )
        self.active.append(row)

    @torch.no_grad()
    def _decode_step(self) -> None:
        assert self.cache is not None and self.mask is not None
        rows = len(self.active)
        idx = torch.tensor([[a.pending] for a in self.active])
        positions = torch.tensor([[a.cache_len] for a in self.active])
        mask = torch.cat((self.mask, torch.ones(rows, 1, dtype=torch.bool)), dim=1)
        logits, _, self.cache = self.model(
            idx, cache=self.cache, positions=positions, key_mask=mask
        )
        self.mask = mask
        for a, tok in zip(self.active, self._sample(logits[:, -1])):
            a.cache_len += 1
            a.out.append(tok)
            a.pending = tok
        self.stats["decode_steps"] += 1
        self.stats["batch_rows"] += rows
        self.stats["generated"] += rows
        self._retire()

    def _retire(self) -> None:
        keep = [i for i, a in enumerate(self.active) if not self._finished(a)]
        for a in self.active:
            if self._finished(a):
                self.results[a.rid] = a.out
        if len(keep) == len(self.active):
            return
        if not keep:
            self.active, self.cache, self.mask = [], None, None
            return
        assert self.cache is not None and self.mask is not None
        sel = torch.tensor(keep)
        self.active = [self.active[i] for i in keep]
        mask = self.mask[sel]
        # Drop leading columns that are padding for every surviving row.
        first = int(mask.any(dim=0).nonzero()[0])
        self.mask = mask[:, first:]
        self.cache = [(k[sel][:, :, first:], v[sel][:, :, first:]) for k, v in self.cache]
