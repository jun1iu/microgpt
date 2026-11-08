import pytest
import torch

from microgpt.model import GPT, GPTConfig, apply_rope, rope_tables

CFG = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_embd=32)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_output_shapes(model):
    idx = torch.randint(0, CFG.vocab_size, (3, 10))
    logits, loss, cache = model(idx)
    assert logits.shape == (3, 10, CFG.vocab_size)
    assert loss is None
    assert len(cache) == CFG.n_layer and cache[0][0].shape == (3, CFG.n_head, 10, 8)


def test_weights_are_tied(model):
    assert model.lm_head.weight is model.tok_emb.weight


def test_no_future_leakage(model):
    """Changing a later token must not change logits at earlier positions."""
    a = torch.randint(0, CFG.vocab_size, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % CFG.vocab_size
    with torch.no_grad():
        la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-6)
    assert not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-6)


def test_kv_cache_matches_full_forward(model):
    """Prefill + one-token-at-a-time decoding == a single full forward pass."""
    idx = torch.randint(0, CFG.vocab_size, (2, 20))
    with torch.no_grad():
        full, _, _ = model(idx)
        logits, _, cache = model(idx[:, :7])
        steps = [logits]
        for t in range(7, 20):
            step, _, cache = model(idx[:, t : t + 1], cache=cache)
            steps.append(step)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_cache_chunked_prefill_matches_full(model):
    """Multi-token continuation exercises the non-trivial causal-mask offset."""
    idx = torch.randint(0, CFG.vocab_size, (1, 18))
    with torch.no_grad():
        full, _, _ = model(idx)
        l1, _, cache = model(idx[:, :6])
        l2, _, _ = model(idx[:, 6:], cache=cache)
    assert torch.allclose(full, torch.cat([l1, l2], dim=1), atol=1e-5)


def test_greedy_generation_identical_with_and_without_cache(model):
    prompt = torch.randint(0, CFG.vocab_size, (1, 5))
    a = model.generate(prompt, 20, temperature=0, use_cache=True)
    b = model.generate(prompt, 20, temperature=0, use_cache=False)
    assert torch.equal(a, b)


def test_generation_past_block_size(model):
    prompt = torch.randint(0, CFG.vocab_size, (1, 5))
    out = model.generate(prompt, CFG.block_size * 2, temperature=1.0, top_k=10)
    assert out.shape == (1, 5 + CFG.block_size * 2)
    assert int(out.max()) < CFG.vocab_size


def test_sequence_longer_than_block_size_is_rejected(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, CFG.block_size + 1, dtype=torch.long))


def test_seeded_sampling_is_reproducible(model):
    prompt = torch.randint(0, CFG.vocab_size, (1, 4))
    runs = [
        model.generate(prompt, 15, top_p=0.9, generator=torch.Generator().manual_seed(7))
        for _ in range(2)
    ]
    assert torch.equal(runs[0], runs[1])


def test_rope_is_norm_preserving():
    cos, sin = rope_tables(16, 40, 10000.0)
    x = torch.randn(2, 3, 40, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_scores_depend_only_on_relative_position():
    """<rope(q, m), rope(k, n)> must equal <rope(q, m+s), rope(k, n+s)>."""
    cos, sin = rope_tables(16, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m: int, n: int) -> torch.Tensor:
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum()

    assert torch.allclose(score(5, 2), score(25, 22), atol=1e-4)
    assert torch.allclose(score(0, 9), score(30, 39), atol=1e-4)
    assert not torch.allclose(score(5, 2), score(5, 3), atol=1e-4)


def test_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    m = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        loss = m(x, y)[1]
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first
