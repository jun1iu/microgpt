"""Decoder-only transformer written from scratch.

Architecture (the Llama recipe, at toy scale):
  * RMSNorm pre-normalisation
  * Rotary position embeddings (RoPE) applied to queries and keys
  * Causal self-attention, written out explicitly (no nn.Transformer), with optional
    grouped-query attention (GQA/MQA) to shrink the KV cache
  * SwiGLU feed-forward network
  * Tied input/output embeddings
  * Per-layer key/value cache for incremental decoding
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from microgpt.sampling import sample_next

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    rope_base: float = 10000.0
    n_kv_head: Optional[int] = None  # None -> multi-head; 1 -> multi-query; else grouped-query

    @property
    def kv_heads(self) -> int:
        return self.n_head if self.n_kv_head is None else self.n_kv_head


def kv_cache_bytes(cache: KVCache) -> int:
    """Total bytes held by a key/value cache."""
    return sum(t.numel() * t.element_size() for kv in cache for t in kv)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(head_dim: int, max_len: int, base: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_len, head_dim // 2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive channel pairs of x (B, H, T, hd) by position-dependent angles.

    Because rotations compose, <rope(q, m), rope(k, n)> depends only on m - n, which is
    how relative position enters attention without any learned position table.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        if cfg.n_head % cfg.kv_heads != 0:
            raise ValueError(f"n_head {cfg.n_head} not divisible by kv heads {cfg.kv_heads}")
        self.n_head = cfg.n_head
        self.n_kv = cfg.kv_heads
        self.head_dim = cfg.n_embd // cfg.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        # One projection produces n_head query heads but only n_kv key/value heads.
        self.qkv = nn.Linear(cfg.n_embd, (cfg.n_head + 2 * self.n_kv) * self.head_dim, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        cos, sin = rope_tables(self.head_dim, cfg.block_size, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        positions: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: (B, T, C). Optional batching arguments, used for variable-length batches:

        positions: (B, T) true position of each new token (so left-padding does not shift RoPE).
        key_mask:  (B, past + T) bool, False for padding columns that must never be attended to.
        """
        B, T, C = x.shape
        hd = self.head_dim
        past = 0 if cache is None else cache[0].size(2)

        q, k, v = self.qkv(x).split([self.n_head * hd, self.n_kv * hd, self.n_kv * hd], dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)  # (B, n_head, T, hd)
        k = k.view(B, T, self.n_kv, hd).transpose(1, 2)  # (B, n_kv,   T, hd)
        v = v.view(B, T, self.n_kv, hd).transpose(1, 2)

        # Positions are absolute, so cached keys keep the rotation they were stored with.
        if positions is None:
            cos, sin = self.cos[past : past + T], self.sin[past : past + T]
        else:
            cos, sin = self.cos[positions].unsqueeze(1), self.sin[positions].unsqueeze(1)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        new_cache = (k, v)  # cached at n_kv heads: this is where GQA saves memory

        # Each group of g query heads shares one key/value head. Instead of copying K/V g times
        # we view q as (B, n_kv, g, T, hd) and let matmul broadcast over the group axis.
        S = k.size(2)  # total keys = past + T
        g = self.n_head // self.n_kv
        q = q.reshape(B, self.n_kv, g, T, hd)
        scores = (q @ k.unsqueeze(2).transpose(-2, -1)) / math.sqrt(hd)  # (B, n_kv, g, T, S)

        # Query i sits at absolute column past + i and may see columns j <= past + i.
        allowed = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        if key_mask is not None:
            # A padded query would otherwise have no visible key (softmax over all -inf = NaN,
            # which would then poison real tokens through v). Let every query see itself.
            own = torch.arange(T, device=x.device)[:, None] + (S - T) == torch.arange(
                S, device=x.device
            )
            allowed = ((allowed[None] & key_mask[:, None, :]) | own[None])[:, None, None]
        scores = scores.masked_fill(~allowed, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v.unsqueeze(2)  # (B, n_kv, g, T, hd)
        out = out.reshape(B, self.n_head, T, hd).transpose(1, 2).reshape(B, T, C)
        return self.proj(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 32 * ((hidden + 31) // 32)
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cache=None, positions=None, key_mask=None):
        a, new_cache = self.attn(self.norm1(x), cache, positions, key_mask)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)
        # GPT-2 style: shrink residual-branch outputs so variance stays flat with depth.
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        positions: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], KVCache]:
        """Return (logits, loss, new_cache). Pass `cache` to decode incrementally.

        `positions` / `key_mask` enable variable-length batches (see microgpt.batching).
        """
        _, T = idx.shape
        past = 0 if cache is None else cache[0][0].size(2)
        if positions is None:
            if past + T > self.cfg.block_size:
                raise ValueError(
                    f"sequence length {past + T} exceeds block_size {self.cfg.block_size}"
                )
        elif int(positions.max()) >= self.cfg.block_size:
            raise ValueError(f"position exceeds block_size {self.cfg.block_size}")

        x = self.tok_emb(idx)
        new_cache: KVCache = []
        for i, block in enumerate(self.blocks):
            x, layer_cache = block(x, None if cache is None else cache[i], positions, key_mask)
            new_cache.append(layer_cache)
        logits = self.lm_head(self.norm_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_cache

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Autoregressively extend idx (B, T). Beyond block_size the window slides.

        With the cache each step feeds one token (O(T) attention work per step); without it
        every step re-runs the whole window (O(T^2)). While the sequence fits in block_size
        both paths give identical tokens (verified in tests). Past that point they differ by
        design: the uncached path always sees the last block_size tokens, whereas the cached
        path drops its cache when full and rebuilds it from the most recent half-window, so
        it pays one re-prefill instead of invalidating positions on every step.
        """
        was_training = self.training
        self.eval()
        block = self.cfg.block_size
        cache: Optional[KVCache] = None
        for _ in range(max_new_tokens):
            if use_cache:
                if cache is not None and cache[0][0].size(2) >= block:
                    cache = None  # window full: rebuild from the tail
                if cache is None:
                    window = idx[:, -(block // 2) :] if idx.size(1) >= block else idx
                    logits, _, cache = self(window, cache=None)
                else:
                    logits, _, cache = self(idx[:, -1:], cache=cache)
            else:
                logits, _, _ = self(idx[:, -block:])
            next_tok = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
            idx = torch.cat((idx, next_tok), dim=1)
        self.train(was_training)
        return idx
