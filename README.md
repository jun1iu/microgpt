# microgpt

A GPT-style language model written from scratch in PyTorch, plus the inference techniques that
matter in production LLM serving: a **KV cache**, **grouped-query attention**, **continuous
batching** over variable-length requests, and **speculative decoding**. Every one of them is
backed by tests that check correctness against a reference, not just "it runs".

No `nn.Transformer`, no Hugging Face. Attention, masking, caching, batching and sampling are all
written out explicitly (~900 lines of library code) so every line can be reasoned about.

## What's inside

| Piece | Where | The interesting part |
|---|---|---|
| Causal attention, RoPE, RMSNorm, SwiGLU, tied embeddings | `model.py` | The Llama recipe at toy scale; mask offset by cache length |
| Grouped-query / multi-query attention | `model.py` | Cache stores only the KV heads; q is regrouped so matmul broadcasts instead of copying K/V |
| KV cache + sliding window | `model.py` | `O(T)` work per generated token instead of `O(T^2)` |
| Continuous batching | `batching.py` | Left-padded shared cache, per-row RoPE positions, key mask, mid-flight admission, column trimming |
| Speculative decoding | `speculative.py` | Exact target distribution (rejection sampling + residual), cache rollback |
| Sampling: temperature / top-k / top-p | `sampling.py` | `logits_to_probs` exposes the exact sampling distribution (greedy = one-hot) |
| Trainer | `train.py` | AdamW with decoupled decay, warmup + cosine LR, grad clipping |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # 49 tests, ~15s on CPU

mkdir -p data && curl -sSL -o data/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python -m microgpt.train --data data/shakespeare.txt --steps 1200 --batch-size 24
python -m microgpt.train --data data/shakespeare.txt --n-layer 1 --n-embd 64 --n-head 2 \
    --steps 2000 --out checkpoints/draft.pt            # small draft model

python -m microgpt.generate --prompt "ROMEO:" --tokens 200 --top-k 40
python -m microgpt.generate --prompt "ROMEO:" --tokens 200 --draft checkpoints/draft.pt  # speculative

python scripts/benchmark.py              # KV cache vs no cache
python scripts/bench_gqa.py              # cache memory: MHA vs GQA vs MQA
python scripts/bench_batching.py         # continuous batching throughput
python scripts/bench_speculative.py      # speculative decoding speedup
```

## How each feature is verified

**KV cache.** Prefill-then-decode reproduces the full forward pass to `1e-5`, including
multi-token continuation (which exercises the offset causal mask). Perturbing future tokens never
changes earlier logits. RoPE preserves norms and its attention scores depend only on `m - n`.

**Grouped-query attention.** A GQA model must equal a multi-head model whose K/V heads are
*copies* of the GQA ones (weights are copied across and logits compared, both full-forward and
token-by-token). Cache bytes are asserted to scale exactly with the number of KV heads.

**Continuous batching.** Requests of different lengths share one cache. The tests require that
greedy output for every request is *identical* to running it alone, across: queueing behind a
full batch, requests joining mid-flight with longer *and* shorter prompts than the current
batch, EOS stopping one row without disturbing others, and the `block_size` limit. A padded row
must not produce NaN (more on that below).

**Speculative decoding.** At temperature 0 the output equals greedy target decoding
token-for-token. For sampling, the test computes the target's exact distribution over the first
two generated tokens and the marginal of the third by brute force on a 5-token vocabulary, then
checks 2,500 speculative samples against it. A power check asserts the draft really is a
different distribution, and I confirmed the test fails when the algorithm is deliberately broken
(always accepting proposals, or resampling from `p` instead of the residual `max(0, p - q)`).

## Results

All numbers were measured with this repo's scripts on an Intel iMac (6 CPU threads, no GPU).
CPU timings are noisy; treat small differences as approximate.

### Training (character-level Tiny Shakespeare)

0.81M parameters (4 layers, 128-dim), 1200 steps, about 5.6 minutes:

| step | train loss | val loss |
|---:|---:|---:|
| 0 | 4.242 | 4.235 |
| 400 | 1.657 | 1.830 |
| 800 | 1.444 | 1.655 |
| 1200 | 1.348 | 1.596 |

Initial loss 4.24 is close to ln 65 = 4.17 (uniform over the 65-character vocabulary), as expected.

### KV cache

| new tokens | no cache (tok/s) | KV cache (tok/s) | speedup | identical output |
|-----------:|-----------------:|-----------------:|--------:|:-----------------:|
| 32 | 158.3 | 194.2 | 1.2x | yes |
| 128 | 98.2 | 174.4 | 1.8x | yes |
| 256 | 74.5 | 198.8 | 2.7x | yes |
| 480 | 34.7 | 167.4 | 4.8x | yes |

Cached throughput stays roughly flat as the sequence grows while uncached throughput decays.

### Grouped-query attention

8 query heads, 6 layers, 256-dim, cache holding 496 tokens:

| KV heads | variant | params | KV cache | vs MHA |
|---:|:--|---:|---:|---:|
| 8 | MHA | 4.84M | 5.8 MiB | 1.00x |
| 4 | GQA | 4.45M | 2.9 MiB | 0.50x |
| 2 | GQA | 4.25M | 1.5 MiB | 0.25x |
| 1 | MQA | 4.16M | 0.7 MiB | 0.12x |

Memory is exact. Decode speed on this CPU (126 / 176 / 136 / 133 tok/s for 8 / 4 / 2 / 1 KV
heads) is not monotonic, which suggests noise, but I did not measure run-to-run variance, so no
speed claim is made. The benefit of GQA shown here is cache size, which matters when serving
long contexts or many concurrent sequences.

### Continuous batching

16 requests (prompts of 8-47 tokens, 64 new tokens each), 6-layer 192-dim model:

| strategy | tokens/s | speedup | mean batch |
|:--|--:|--:|--:|
| one at a time | 240 | 1.0x | 1.0 |
| continuous, max_batch=2 | 303 | 1.3x | 2.0 |
| continuous, max_batch=4 | 575 | 2.4x | 4.0 |
| continuous, max_batch=8 | 1144 | 4.8x | 8.0 |
| continuous, max_batch=16 | 1767 | 7.4x | 16.0 |

Caveat: every request here asks for the same number of tokens, so the batch stays full. Real
traffic with mixed lengths would see lower occupancy; the scheduler refills slots as rows finish.

### Speculative decoding

Target: the 0.81M model (val loss 1.60). Draft: 57.6K parameters, 1 layer (val loss 1.81).
Temperature 0.8, 100 new tokens, 12 runs per row:

| method | s / run | speedup | acceptance | tokens / target pass |
|:--|--:|--:|--:|--:|
| target only | 0.36 | 1.00x | - | 1.0 |
| speculative, gamma=2 | 0.26 | 1.38x | 64% | 2.26 |
| speculative, gamma=4 | 0.22 | 1.65x | 50% | 2.96 |
| speculative, gamma=6 | 0.22 | 1.68x | 44% | 3.59 |
| speculative, gamma=8 | 0.25 | 1.42x | 32% | 3.50 |

The speedup peaks at moderate `gamma` and then falls: more proposals per round means each is less
likely to be accepted, so the draft's extra work stops paying for itself. The output
distribution is the target's in every row (see verification above), so this is a pure speed
trade-off. Wall-clock gains depend on the draft being much cheaper than the target; with a
GPU-scale target the same acceptance rates would give a larger speedup.

## Pitfalls hit (and avoided) while building this

* **Sliding window.** An early version claimed cached and uncached generation were identical.
  They are, but only while the text fits in `block_size`; past it the two paths slide the window
  differently. Caught by seeded samples diverging at token 135; the docstring and README now say so.
* **NaN from padding.** In a left-padded batch, a padding query has no visible key, so softmax over
  all `-inf` gives NaN, and NaN * 0 in the value product poisons real tokens. Fix: every query may
  always attend to itself (its output is discarded anyway). Covered by a finite-logits assertion.
* **Ties in parity tests (avoided).** The default init gives near-uniform logits, where greedy
  argmax could flip on float noise between code paths. Parity tests therefore use a wider init
  (`tests/helpers.py`) so they compare confident outputs rather than ties.
* **Wrong test, not wrong code.** One batching test failed because it sampled cache width *after*
  a row had already been retired in the same step; the engine was right, the assertion was not.

## Limitations

* Past `block_size`, the single-sequence cache path drops and rebuilds its cache from the latest
  half-window; the batching engine and speculative decoding instead require sequences to fit in
  `block_size`.
* Speculative decoding handles one sequence at a time and does not combine with the batch engine.
* The cache grows by `torch.cat` each step (simple and correct, but it reallocates); a production
  system would use preallocated or paged blocks.
* Attention is the plain `O(T^2)`-memory formulation: no FlashAttention.
* Character-level and small-scale: the point is correctness and clarity, not sample quality.

## Roadmap

- [ ] BPE tokenizer trained from scratch, with round-trip tests
- [ ] Paged KV cache (fixed-size blocks, block table) to remove the `torch.cat` reallocation
- [ ] Speculative decoding inside the batch engine
- [ ] Load a public checkpoint into this architecture and match Hugging Face logits
