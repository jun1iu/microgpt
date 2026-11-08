"""Character-level tokenizer and batching."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


class CharTokenizer:
    def __init__(self, chars: Sequence[str]) -> None:
        self.chars: List[str] = list(chars)
        self._stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self._stoi[c] for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.chars[i] for i in ids)


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random (x, y) windows where y is x shifted left by one token."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts])
    return x, y
