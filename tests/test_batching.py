import pytest
import torch
from helpers import make_model

from microgpt.batching import BatchEngine
from microgpt.model import GPTConfig

CFG = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_embd=32, n_kv_head=2)


@pytest.fixture(scope="module")
def model():
    return make_model(CFG, seed=3)


def alone(model, prompt, n):
    out = model.generate(torch.tensor([prompt]), n, temperature=0)
    return out[0, len(prompt) :].tolist()


def prompt_of(length, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, CFG.vocab_size, (length,), generator=g).tolist()


def test_left_padded_forward_matches_unpadded(model):
    """The core mechanic: padding + explicit positions + key mask is invisible to real tokens."""
    prompts = [prompt_of(n, n) for n in (3, 9, 6)]
    width = max(map(len, prompts))
    idx = torch.zeros(3, width, dtype=torch.long)
    key_mask = torch.zeros(3, width, dtype=torch.bool)
    positions = torch.zeros(3, width, dtype=torch.long)
    for r, p in enumerate(prompts):
        idx[r, width - len(p) :] = torch.tensor(p)
        key_mask[r, width - len(p) :] = True
        positions[r, width - len(p) :] = torch.arange(len(p))
    with torch.no_grad():
        batched = model(idx, positions=positions, key_mask=key_mask)[0]
        for r, p in enumerate(prompts):
            solo = model(torch.tensor([p]))[0][0]
            assert torch.allclose(batched[r, width - len(p) :], solo, atol=1e-5)
    assert torch.isfinite(batched).all()  # padded rows must not produce NaN


def test_engine_matches_individual_generation(model):
    """5 requests, batch of 3: forces queueing and mid-flight admission."""
    reqs = [(prompt_of(n, 10 + n), m) for n, m in [(3, 12), (8, 5), (5, 14), (1, 9), (7, 10)]]
    eng = BatchEngine(model, max_batch=3)
    ids = [eng.submit(p, m) for p, m in reqs]
    results = eng.run()
    for rid, (p, m) in zip(ids, reqs):
        assert results[rid] == alone(model, p, m)
    assert eng.cache is None and not eng.active  # fully drained
    assert eng.mean_batch_size > 1.5  # it actually batched


def test_requests_joining_mid_flight(model):
    a, b, c = prompt_of(4, 1), prompt_of(10, 2), prompt_of(2, 3)
    eng = BatchEngine(model, max_batch=4)
    ra = eng.submit(a, 15)
    for _ in range(4):
        eng.step()
    rb = eng.submit(b, 8)  # longer prompt than the current batch width
    eng.step()
    rc = eng.submit(c, 6)  # shorter prompt than the current batch width
    results = eng.run()
    assert results[ra] == alone(model, a, 15)
    assert results[rb] == alone(model, b, 8)
    assert results[rc] == alone(model, c, 6)


def test_eos_stops_a_sequence_without_disturbing_others(model):
    p, q = prompt_of(5, 21), prompt_of(6, 22)
    free_run = alone(model, p, 10)
    eos = free_run[3]
    cut = free_run.index(eos) + 1  # first occurrence, inclusive
    eng = BatchEngine(model, max_batch=2)
    rp = eng.submit(p, 10, eos_id=eos)
    rq = eng.submit(q, 10)
    results = eng.run()
    assert results[rp] == free_run[:cut]
    assert results[rq] == alone(model, q, 10)


def test_sequence_stops_at_block_size(model):
    p = prompt_of(30, 5)  # 2 free positions + the first token from the prefill
    eng = BatchEngine(model)
    rid = eng.submit(p, 10)
    out = eng.run()[rid]
    assert len(out) == 3
    assert out == alone(model, p, 3)


def test_finished_rows_release_cache_columns(model):
    long_p, short_p = prompt_of(20, 7), prompt_of(2, 8)
    eng = BatchEngine(model, max_batch=2)
    eng.submit(long_p, 5)  # leaves after 4 decode steps (first token comes from the prefill)
    eng.submit(short_p, 10)
    widths = []
    while eng.step():
        if eng.mask is not None:
            widths.append(eng.mask.size(1))
    assert max(widths) >= 20 and widths[-1] < 20  # width shrank after the long row left


def test_submit_validation(model):
    eng = BatchEngine(model)
    with pytest.raises(ValueError):
        eng.submit([], 3)
    with pytest.raises(ValueError):
        eng.submit([1] * (CFG.block_size + 1), 3)
    with pytest.raises(ValueError):
        eng.submit([1], 0)
