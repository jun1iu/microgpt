import itertools

import pytest
import torch
from helpers import make_model

from microgpt.model import GPTConfig
from microgpt.sampling import logits_to_probs
from microgpt.speculative import speculative_generate

V = 50
TARGET_CFG = GPTConfig(vocab_size=V, block_size=64, n_layer=2, n_head=4, n_embd=32)
DRAFT_CFG = GPTConfig(vocab_size=V, block_size=64, n_layer=1, n_head=2, n_embd=16)


@pytest.fixture(scope="module")
def target():
    return make_model(TARGET_CFG, seed=1)


@pytest.fixture(scope="module")
def draft():
    return make_model(DRAFT_CFG, seed=2)


def prompt_of(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (1, n), generator=g)


@pytest.mark.parametrize("gamma", [1, 3, 6])
def test_greedy_speculation_equals_greedy_target(target, draft, gamma):
    """At temperature 0 the output must be token-for-token the target's greedy output."""
    prompt = prompt_of(5)
    expected = target.generate(prompt, 30, temperature=0)
    got, _ = speculative_generate(target, draft, prompt, 30, gamma=gamma, temperature=0)
    assert torch.equal(got, expected)


def test_a_perfect_draft_is_always_accepted(target):
    out, stats = speculative_generate(
        target, target, prompt_of(4), 40, gamma=4, temperature=1.0,
        generator=torch.Generator().manual_seed(0),
    )  # fmt: skip
    assert out.shape == (1, 44)
    assert stats.acceptance_rate > 0.97
    assert stats.tokens_per_target_call > 3.5  # ~gamma + 1 per target forward


def test_stats_are_consistent(target, draft):
    _, stats = speculative_generate(target, draft, prompt_of(3), 25, gamma=4, temperature=1.0)
    assert stats.new_tokens == 25
    assert stats.target_calls == stats.rounds
    assert 0 <= stats.accepted <= stats.proposed
    assert stats.target_calls < 25  # fewer target passes than tokens, even with a poor draft


def test_seeded_runs_are_reproducible(target, draft):
    runs = [
        speculative_generate(
            target, draft, prompt_of(3), 20, temperature=0.9, top_p=0.95,
            generator=torch.Generator().manual_seed(11),
        )[0]
        for _ in range(2)
    ]  # fmt: skip
    assert torch.equal(runs[0], runs[1])


def test_output_distribution_is_exactly_the_targets():
    """The defining guarantee: speculation changes speed, never the distribution.

    Tiny vocabulary so the exact target distribution over the first two tokens (25 outcomes)
    and the marginal of the third can be computed by brute force and compared with samples.
    """
    v = 5
    t = make_model(GPTConfig(vocab_size=v, block_size=16, n_layer=2, n_head=2, n_embd=16), 4, 0.4)
    d = make_model(GPTConfig(vocab_size=v, block_size=16, n_layer=1, n_head=2, n_embd=16), 9, 0.4)
    prompt = torch.tensor([[1, 3]])

    def dist(model, tokens):
        with torch.no_grad():
            return logits_to_probs(model(torch.tensor([tokens]))[0][:, -1], 1.0)[0]

    def exact(model):
        joint = torch.zeros(v, v)
        third = torch.zeros(v)
        for a, b in itertools.product(range(v), repeat=2):
            w = dist(model, [1, 3])[a] * dist(model, [1, 3, a])[b]
            joint[a, b] = w
            third += w * dist(model, [1, 3, a, b])
        return joint, third

    joint_t, third_t = exact(t)
    joint_d, _ = exact(d)
    # Power check: the draft really is a different distribution, so "just sample the draft"
    # (the bug this test exists to catch) would be detected below.
    assert (joint_t - joint_d).abs().max() > 0.04

    n = 2500
    g = torch.Generator().manual_seed(123)
    joint = torch.zeros(v, v)
    third = torch.zeros(v)
    for _ in range(n):
        out, _ = speculative_generate(t, d, prompt, 3, gamma=4, temperature=1.0, generator=g)
        a, b, c = out[0, 2:].tolist()
        joint[a, b] += 1 / n
        third[c] += 1 / n
    assert (joint - joint_t).abs().max() < 0.03
    assert (third - third_t).abs().max() < 0.03


def test_top_k_support_is_respected(target, draft):
    """Every emitted token must lie in the target's own top-k set given its prefix."""
    g = torch.Generator().manual_seed(5)
    for trial in range(40):
        out, _ = speculative_generate(
            target, draft, prompt_of(4, trial), 6, gamma=3, temperature=1.0, top_k=3, generator=g
        )
        for pos in range(4, 10):
            with torch.no_grad():
                top = target(out[:, :pos])[0][0, -1].topk(3).indices.tolist()
            assert int(out[0, pos]) in top


def test_validation(target, draft):
    with pytest.raises(ValueError):
        speculative_generate(target, draft, prompt_of(3).repeat(2, 1), 5)  # batch > 1
    with pytest.raises(ValueError):
        speculative_generate(target, draft, prompt_of(60), 10)  # exceeds block_size
    with pytest.raises(ValueError):
        speculative_generate(target, draft, prompt_of(3), 5, gamma=0)
    other = make_model(GPTConfig(vocab_size=V + 1, block_size=64, n_layer=1, n_head=2, n_embd=16))
    with pytest.raises(ValueError):
        speculative_generate(target, other, prompt_of(3), 5)


def test_training_mode_is_restored(target, draft):
    target.train()
    speculative_generate(target, draft, prompt_of(3), 5, temperature=0)
    assert target.training and not draft.training
    target.eval()


def test_temperature_zero_probs_are_one_hot():
    probs = logits_to_probs(torch.tensor([[0.1, 3.0, -2.0]]), temperature=0)
    assert probs.tolist() == [[0.0, 1.0, 0.0]]
