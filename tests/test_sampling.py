import pytest
import torch

from microgpt.data import CharTokenizer, get_batch
from microgpt.sampling import filter_top_k, filter_top_p, sample_next


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.0]])
    assert sample_next(logits, temperature=0).flatten().tolist() == [1, 0]


def test_negative_temperature_rejected():
    with pytest.raises(ValueError):
        sample_next(torch.zeros(1, 3), temperature=-1.0)


def test_top_k_keeps_exactly_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    kept = torch.isfinite(filter_top_k(logits, 2))
    assert kept.tolist() == [[False, True, False, True, False]]


def test_top_p_keeps_smallest_nucleus():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    out = filter_top_p(probs.log(), 0.7)
    assert torch.isfinite(out).tolist() == [[True, True, False, False]]


def test_top_p_always_keeps_the_top_token():
    out = filter_top_p(torch.tensor([[10.0, 0.0, 0.0]]), 0.01)
    assert torch.isfinite(out).tolist() == [[True, False, False]]


def test_samples_respect_top_k():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]]).expand(500, -1)
    draws = sample_next(logits, top_k=2, generator=g)
    assert set(draws.flatten().tolist()) <= {3, 4}


def test_sampling_matches_softmax_distribution():
    g = torch.Generator().manual_seed(0)
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]])).expand(20000, -1)
    freq = torch.bincount(sample_next(logits, generator=g).flatten(), minlength=3) / 20000
    assert torch.allclose(freq, torch.tensor([0.7, 0.2, 0.1]), atol=0.02)


def test_tokenizer_roundtrip():
    text = "hello, world\n"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == len(set(text))


def test_get_batch_targets_are_shifted_inputs():
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)
