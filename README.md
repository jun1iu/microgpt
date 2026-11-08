# microgpt

A GPT-style language model written from scratch in PyTorch, with the pieces that matter in
production LLM systems: rotary position embeddings, RMSNorm, SwiGLU, an incremental-decoding
**KV cache**, nucleus sampling, and a test suite that proves each piece is correct.

No `nn.Transformer`, no Hugging Face. Attention, masking, caching and sampling are all
written out explicitly (~250 lines of model code) so every line can be reasoned about.

## What's inside

| Piece | Where | Why it's here |
|---|---|---|
| Causal self-attention with explicit mask | `model.py` | Mask is offset by the cache length, which is the classic place incremental decoding goes wrong |
| RoPE | `model.py` | Relative position without a learned table; cached keys keep their absolute rotation |
| RMSNorm + SwiGLU, tied embeddings | `model.py` | The Llama-style recipe at toy scale |
| KV cache + sliding window | `model.py` | `O(T)` work per generated token instead of `O(T^2)` |
| Temperature / top-k / top-p | `sampling.py` | Written from first principles, tested against analytic distributions |
| Trainer | `train.py` | AdamW with decoupled weight decay (no decay on norms), warmup + cosine LR, grad clipping |
| Benchmark | `scripts/benchmark.py` | Measures the cache speedup instead of asserting it |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                  # 21 tests, ~3s on CPU

mkdir -p data && curl -sSL -o data/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python -m microgpt.train --data data/shakespeare.txt --steps 1200 --batch-size 24
python -m microgpt.generate --prompt "ROMEO:" --tokens 200 --top-k 40
python scripts/benchmark.py
```

## Correctness: what the tests actually prove

* **No future leakage** — perturbing tokens at positions >= 8 leaves logits at positions < 8 unchanged.
* **KV cache == full forward** — prefill then one-token-at-a-time decoding reproduces the full
  forward pass to `1e-5`; so does multi-token continuation (which exercises the offset causal mask).
* **Greedy generation is identical with and without the cache** (while within the context window).
* **RoPE** preserves vector norms, and `<rope(q, m), rope(k, n)>` depends only on `m - n`.
* **Sampling** — top-k and top-p keep exactly the intended tokens; empirical frequencies match
  the softmax distribution; seeded runs are reproducible.
* **Learning** — the model can overfit a tiny batch (loss drops by more than 70%).

## Results

Measured on an Intel iMac (6 CPU threads, no GPU). Numbers are from this repo's own scripts.

**Training** — 0.81M parameters, 4 layers, 128-dim, character-level Tiny Shakespeare,
1200 steps (~5.6 min on CPU):

| step | train loss | val loss |
|---:|---:|---:|
| 0 | 4.242 | 4.235 |
| 400 | 1.657 | 1.830 |
| 800 | 1.444 | 1.655 |
| 1200 | 1.348 | 1.596 |

(Initial loss 4.24 ≈ ln 65, i.e. uniform over the 65-character vocabulary, as expected.)

**KV-cache decoding** — 2.68M-parameter model, greedy, 16-token prompt:

| new tokens | no cache (tok/s) | KV cache (tok/s) | speedup | identical output |
|-----------:|-----------------:|-----------------:|--------:|:-----------------:|
| 32 | 158.3 | 194.2 | 1.2x | yes |
| 64 | 166.2 | 259.3 | 1.6x | yes |
| 128 | 98.2 | 174.4 | 1.8x | yes |
| 256 | 74.5 | 198.8 | 2.7x | yes |
| 480 | 34.7 | 167.4 | 4.8x | yes |

Cached throughput stays roughly flat as the sequence grows while uncached throughput decays,
which is the `O(T)` vs `O(T^2)` difference showing up in practice. Timings on a shared CPU are
noisy, so treat the small-length rows as approximate.

## Design notes and known limitations

* **Sliding window past `block_size`.** The uncached path always sees the last `block_size`
  tokens. The cached path drops its cache when full and rebuilds it from the latest half-window.
  Outputs are therefore identical only while the sequence fits in the window; beyond that they
  differ by design. (RoPE makes true cache eviction possible, but keys would need re-rotation.)
* **Attention is `O(T^2)` memory** per layer: no FlashAttention, no grouped-query attention.
  Both are natural next steps and would also shrink the KV cache.
* **Character-level, small scale.** The point is correctness and clarity, not sample quality.
  Samples are Shakespeare-flavoured but not coherent.

## Roadmap

- [ ] Grouped-query attention (smaller KV cache) with a parity test against full multi-head
- [ ] Batched generation with per-sequence cache lengths
- [ ] Speculative decoding with a draft model, verified to preserve the target distribution
- [ ] BPE tokenizer
