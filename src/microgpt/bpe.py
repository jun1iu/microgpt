"""Byte-level BPE tokenizer, trained and applied from scratch.

Design (the GPT-2 family recipe, minus the heavy dependencies):
  * The base vocabulary is the 256 byte values, so *any* string round-trips losslessly:
    there is no "unknown token".
  * Text is first split into chunks (words, numbers, punctuation runs, whitespace) with a regex,
    and merges never cross chunk boundaries. This stops the tokenizer from learning junk like
    "dog." as one token and keeps training fast (we count unique chunks, weighted by frequency).
  * Training repeatedly merges the most frequent adjacent pair into a new token. Pair counts are
    updated incrementally: only the chunks that contain the merged pair are touched.
  * Encoding applies merges in the order they were learned (lowest rank first).
  * Ties are broken by the smallest pair, so training is fully deterministic.
  * Optional special tokens (e.g. "<|endoftext|>") get ids after the merges and are never produced
    by splitting ordinary text.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Pair = Tuple[int, int]

# Contractions | letters | digits | punctuation/symbols (incl. "_") | whitespace.
# Every character matches exactly one alternative, so the chunks always re-join to the input.
PRETOKENIZE = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> List[str]:
    return PRETOKENIZE.findall(text)


def merge_pair(ids: List[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every non-overlapping, left-to-right occurrence of `pair` with `new_id`."""
    out: List[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _pair_multiset(ids: Sequence[int]) -> Counter:
    return Counter(zip(ids, ids[1:]))


class BPETokenizer:
    def __init__(self, merges: Sequence[Pair] = (), special_tokens: Sequence[str] = ()) -> None:
        self.merges: List[Pair] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.special_tokens: List[str] = list(special_tokens)
        self._rank: Dict[Pair, int] = {m: i for i, m in enumerate(self.merges)}
        self._vocab: List[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._vocab.append(self._vocab[a] + self._vocab[b])
        self._special_id = {s: len(self._vocab) + i for i, s in enumerate(self.special_tokens)}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls, text: str, vocab_size: int, special_tokens: Sequence[str] = ()
    ) -> BPETokenizer:
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(
                f"vocab_size {vocab_size} is too small: need at least "
                f"{256 + len(special_tokens)} (256 bytes + {len(special_tokens)} special tokens)"
            )
        freq = Counter(pretokenize(text))
        words: List[List[int]] = [list(w.encode("utf-8")) for w in freq]
        weights: List[int] = list(freq.values())

        counts: Dict[Pair, int] = defaultdict(int)
        where: Dict[Pair, Set[int]] = defaultdict(set)
        for wi, (w, c) in enumerate(zip(words, weights)):
            for pair, k in _pair_multiset(w).items():
                counts[pair] += c * k
                where[pair].add(wi)

        merges: List[Pair] = []
        while len(merges) < n_merges and counts:
            # Most frequent pair; ties go to the numerically smallest pair (deterministic).
            best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            new_id = 256 + len(merges)
            merges.append(best)
            for wi in where.pop(best):
                w, c = words[wi], weights[wi]
                for pair, k in _pair_multiset(w).items():  # retract this word's old pairs
                    counts[pair] -= c * k
                    if counts[pair] == 0:
                        del counts[pair]
                w = words[wi] = merge_pair(w, best, new_id)
                for pair, k in _pair_multiset(w).items():  # add its new pairs
                    counts[pair] += c * k
                    where[pair].add(wi)
            counts.pop(best, None)
        return cls(merges, special_tokens)

    # ------------------------------------------------------------------ properties
    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + len(self.special_tokens)

    # ------------------------------------------------------------------ encode / decode
    def _encode_chunk(self, chunk: str) -> List[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                rank = self._rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            ids = merge_pair(ids, best, 256 + best_rank)  # type: ignore[operator]
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> List[int]:
        """Encode text, treating special-token strings as plain text."""
        out: List[int] = []
        for chunk in pretokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        if not (allow_special and self._special_re):
            return self.encode_ordinary(text)
        out: List[int] = []
        for part in self._special_re.split(text):
            if part in self._special_id:
                out.append(self._special_id[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """Bytes -> UTF-8. A token sequence cut mid-character decodes to U+FFFD, not an error."""
        special_by_id = {i: s for s, i in self._special_id.items()}
        parts: List[bytes] = []
        for i in ids:
            if 0 <= i < len(self._vocab):
                parts.append(self._vocab[i])
            elif i in special_by_id:
                parts.append(special_by_id[i].encode("utf-8"))
            else:
                raise ValueError(f"token id {i} is outside the vocabulary (size {self.vocab_size})")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "type": "bpe",
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BPETokenizer:
        return cls([tuple(m) for m in d["merges"]], d.get("special_tokens", []))

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
