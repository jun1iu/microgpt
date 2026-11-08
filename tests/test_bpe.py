import json
import random
import sys
from collections import Counter

import pytest
import torch

from microgpt.bpe import BPETokenizer, merge_pair, pretokenize
from microgpt.data import CharTokenizer
from microgpt.generate import load_checkpoint
from microgpt.model import GPT, GPTConfig

POOL = list("abcdefg hij.,!?'\n\t_0123456789  ") + ["é", "ü", "日", "本", "😀", "—", "ß"]
PARAGRAPH = (
    "the quick brown fox jumps over the lazy dog. the dog barks; the fox runs away! "
    "it's 2024 and we're testing tokenizers, aren't we? yes, they'll work. "
)


def random_text(seed: int, n: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(POOL) for _ in range(n))


# --------------------------------------------------------------------------- references
def ref_train(text: str, n_merges: int):
    """Naive BPE: recount every pair from scratch each iteration. Slow but obviously correct."""
    freq = Counter(pretokenize(text))
    seqs = {w: list(w.encode("utf-8")) for w in freq}
    merges = []
    for _ in range(n_merges):
        counts: Counter = Counter()
        for w, c in freq.items():
            for pair in zip(seqs[w], seqs[w][1:]):
                counts[pair] += c
        if not counts:
            break
        best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        merges.append(best)
        for w in seqs:
            seqs[w] = merge_pair(seqs[w], best, 255 + len(merges))
    return merges


def ref_encode(chunk: str, merges):
    """Apply every merge, in the order it was learned, to the whole chunk."""
    ids = list(chunk.encode("utf-8"))
    for rank, pair in enumerate(merges):
        ids = merge_pair(ids, pair, 256 + rank)
    return ids


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(PARAGRAPH * 30 + random_text(0, 1500), 256 + 80)


# --------------------------------------------------------------------------- basics
def test_merge_pair_is_non_overlapping_left_to_right():
    assert merge_pair([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_pair([1, 1, 1, 1], (1, 1), 9) == [9, 9]
    assert merge_pair([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert merge_pair([], (1, 2), 9) == []


def test_byte_vocabulary_roundtrips_anything_with_no_merges():
    t = BPETokenizer()
    assert t.vocab_size == 256
    for s in ["", "hello", "héllo wörld 日本語 😀\n\t", "\x00\x01 control", "a" * 1000]:
        assert t.decode(t.encode(s)) == s
    assert t.encode("é") == list("é".encode())


def test_classic_wikipedia_example():
    t = BPETokenizer.train("aaabdaaabac", 259)
    assert t.merges == [(97, 97), (97, 98), (256, 257)]  # aa, ab, (aa)(ab)
    assert t.encode("aaabdaaabac") == [258, 100, 258, 97, 99]
    assert t.decode([258, 100, 258, 97, 99]) == "aaabdaaabac"


def test_training_is_deterministic():
    text = PARAGRAPH * 5
    assert BPETokenizer.train(text, 300).merges == BPETokenizer.train(text, 300).merges


def test_training_stops_when_nothing_is_left_to_merge():
    t = BPETokenizer.train("ab", 400)
    assert t.merges == [(97, 98)] and t.vocab_size == 257


# --------------------------------------------------------------------------- vs references
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_incremental_training_matches_naive_reference(seed):
    text = PARAGRAPH * 3 + random_text(seed, 2500)
    assert BPETokenizer.train(text, 256 + 70).merges == ref_train(text, 70)


def test_encoding_matches_reference_merge_order(tok):
    for seed in range(60):
        s = random_text(seed, 80) + PARAGRAPH[: seed + 5]
        expected = [i for chunk in pretokenize(s) for i in ref_encode(chunk, tok.merges)]
        assert tok.encode(s) == expected


# --------------------------------------------------------------------------- losslessness
def test_overlapping_merges_resolve_by_learned_order():
    """When two merges compete for the same byte, the earlier-learned one must win."""
    t = BPETokenizer(merges=[(97, 98), (98, 99)])  # ab = 256 (rank 0), bc = 257 (rank 1)
    assert t.encode("abc") == [256, 99]  # NOT [97, 257]
    t2 = BPETokenizer(merges=[(98, 99), (97, 98)])  # reversed priority
    assert t2.encode("abc") == [97, 256]
    assert t.decode(t.encode("abcabc")) == "abcabc"


def test_roundtrip_random_unicode(tok):
    for seed in range(400):
        s = random_text(seed, random.Random(seed).randint(0, 70))
        assert tok.decode(tok.encode(s)) == s


def test_pretokenizer_is_lossless_and_splits_sensibly():
    for seed in range(500):
        s = random_text(seed, 60)
        assert "".join(pretokenize(s)) == s
    assert pretokenize("Hello, world!  It's 42_x") == [
        "Hello", ",", " world", "!", " ", " It", "'s", " 42", "_", "x",
    ]  # fmt: skip


def test_merges_never_cross_chunk_boundaries():
    t = BPETokenizer.train("the cat sat on the mat. " * 200, 256 + 100)
    for rank in range(len(t.merges)):
        token = t._vocab[256 + rank]
        assert b" " not in token[1:], token  # a space may only *start* a token


def test_tokenizer_compresses_text(tok):
    ids = tok.encode(PARAGRAPH * 3)
    assert len(PARAGRAPH * 3) / len(ids) > 1.8


# --------------------------------------------------------------------------- special tokens
def test_special_tokens():
    eot = "<|endoftext|>"
    t = BPETokenizer.train(PARAGRAPH * 10, 256 + 20 + 1, special_tokens=[eot])
    assert t.vocab_size == 256 + 20 + 1
    text = f"the fox{eot}the dog"
    ids = t.encode(text)
    assert ids.count(t.vocab_size - 1) == 1
    assert t.decode(ids) == text
    plain = t.encode(text, allow_special=False)  # treated as ordinary characters
    assert t.vocab_size - 1 not in plain and t.decode(plain) == text
    assert t.encode(eot) == [t.vocab_size - 1]


# --------------------------------------------------------------------------- persistence / errors
def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(str(path))
    loaded = BPETokenizer.load(str(path))
    assert loaded.merges == tok.merges and json.loads(path.read_text())["type"] == "bpe"
    s = PARAGRAPH + random_text(3, 100)
    assert loaded.encode(s) == tok.encode(s)


def test_errors():
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", 255)
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", 256, special_tokens=["<x>"])
    t = BPETokenizer()
    for bad in (256, -1):
        with pytest.raises(ValueError):
            t.decode([bad])


def test_decoding_a_split_character_does_not_crash():
    t = BPETokenizer()
    assert t.decode([0xE6]) == "�"  # first byte of a 3-byte character
    assert t.decode(list("日".encode())) == "日"


# --------------------------------------------------------------------------- integration
TINY = dict(n_layer=1, n_head=2, n_embd=16)


def test_train_cli_with_bpe_then_generate(tmp_path, monkeypatch):
    from microgpt import train

    data = tmp_path / "text.txt"
    data.write_text(PARAGRAPH * 40, encoding="utf-8")
    out = tmp_path / "m.pt"
    argv = [
        "train", "--data", str(data), "--out", str(out), "--tokenizer", "bpe",
        "--vocab-size", "300", "--steps", "3", "--batch-size", "2", "--block-size", "16",
        "--n-layer", "1", "--n-embd", "16", "--n-head", "2", "--eval-every", "3",
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    train.main()

    model, loaded = load_checkpoint(str(out))
    assert isinstance(loaded, BPETokenizer)
    assert model.cfg.vocab_size == loaded.vocab_size == 300
    out_ids = model.generate(torch.tensor([loaded.encode("the ")]), 6, temperature=0)
    assert isinstance(loaded.decode(out_ids[0].tolist()), str)


def test_legacy_char_checkpoint_still_loads(tmp_path):
    chars = list("abc ")
    cfg = GPTConfig(vocab_size=len(chars), block_size=16, **TINY)
    path = tmp_path / "old.pt"
    torch.save({"config": cfg.__dict__, "model": GPT(cfg).state_dict(), "chars": chars}, path)
    _, loaded = load_checkpoint(str(path))
    assert isinstance(loaded, CharTokenizer) and loaded.decode(loaded.encode("ab c")) == "ab c"
