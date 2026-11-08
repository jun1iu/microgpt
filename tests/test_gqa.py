import pytest
import torch
from helpers import make_model

from microgpt.model import GPT, GPTConfig, kv_cache_bytes

BASE = dict(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_embd=32)


def _cfg(n_kv_head):
    return GPTConfig(**BASE, n_kv_head=n_kv_head)


def test_cache_shrinks_in_proportion_to_kv_heads():
    idx = torch.randint(0, 50, (2, 16))
    sizes = {}
    for n_kv in (4, 2, 1):
        _, _, cache = make_model(_cfg(n_kv))(idx)
        assert cache[0][0].shape == (2, n_kv, 16, 8)
        sizes[n_kv] = kv_cache_bytes(cache)
    assert sizes[2] * 2 == sizes[4]
    assert sizes[1] * 4 == sizes[4]


def test_fewer_kv_heads_means_fewer_parameters():
    assert GPT(_cfg(2)).num_params() < GPT(_cfg(None)).num_params()


@pytest.mark.parametrize("n_kv", [2, 1])
def test_gqa_equals_mha_with_duplicated_kv_heads(n_kv):
    """GQA must be exactly multi-head attention whose K/V heads are repeated per group.

    Build an MHA model, copy the GQA weights into it with each K/V head duplicated across its
    query group, and require identical logits (full forward and cached decoding).
    """
    gqa = make_model(_cfg(n_kv), seed=1)
    mha = make_model(_cfg(None), seed=2)
    g, hd, C = 4 // n_kv, 8, 32

    state = {k: v.clone() for k, v in gqa.state_dict().items()}
    for name in list(state):
        if name.endswith("attn.qkv.weight"):
            w = state[name]
            q, k, v = w.split([4 * hd, n_kv * hd, n_kv * hd], dim=0)
            dup = lambda t: t.view(n_kv, hd, C).repeat_interleave(g, dim=0).reshape(4 * hd, C)  # noqa: E731
            state[name] = torch.cat([q, dup(k), dup(v)], dim=0)
    mha.load_state_dict(state)

    idx = torch.randint(0, 50, (2, 20))
    with torch.no_grad():
        full_g, full_m = gqa(idx)[0], mha(idx)[0]
        assert torch.allclose(full_g, full_m, atol=1e-5)

        _, _, cg = gqa(idx[:, :9])
        _, _, cm = mha(idx[:, :9])
        for t in range(9, 20):
            lg, _, cg = gqa(idx[:, t : t + 1], cache=cg)
            lm, _, cm = mha(idx[:, t : t + 1], cache=cm)
            assert torch.allclose(lg, lm, atol=1e-5)
            assert torch.allclose(lg[:, 0], full_g[:, t], atol=1e-5)


@pytest.mark.parametrize("n_kv", [None, 2, 1])
def test_cached_decoding_matches_full_forward(n_kv):
    model = make_model(_cfg(n_kv))
    idx = torch.randint(0, 50, (2, 18))
    with torch.no_grad():
        full = model(idx)[0]
        logits, _, cache = model(idx[:, :5])
        steps = [logits]
        for t in range(5, 18):
            step, _, cache = model(idx[:, t : t + 1], cache=cache)
            steps.append(step)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_no_future_leakage_with_gqa():
    model = make_model(_cfg(2))
    a = torch.randint(0, 50, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 50
    with torch.no_grad():
        la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-6)


def test_invalid_group_size_is_rejected():
    with pytest.raises(ValueError):
        GPT(_cfg(3))  # 4 query heads cannot be split into 3 groups


def test_gqa_model_can_learn():
    torch.manual_seed(0)
    m = GPT(_cfg(2))
    x = torch.randint(0, 50, (4, 16))
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
