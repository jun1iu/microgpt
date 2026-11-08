"""Train a character-level GPT on a text file."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from microgpt.bpe import BPETokenizer
from microgpt.data import CharTokenizer, get_batch
from microgpt.model import GPT, GPTConfig


def lr_at(step: int, max_steps: int, max_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay to min_ratio * max_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, batch_size: int, iters: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch_size, model.cfg.block_size)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="path to a UTF-8 text file")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--vocab-size", type=int, default=512, help="BPE vocabulary size (incl. bytes)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-kv-head", type=int, default=None, help="KV heads (GQA); default = n-head")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    text = Path(args.data).read_text(encoding="utf-8")
    cut = int(0.9 * len(text))
    train_text, val_text = text[:cut], text[cut:]
    if args.tokenizer == "bpe":
        tok = BPETokenizer.train(train_text, args.vocab_size)  # merges learned on train split only
    else:
        tok = CharTokenizer.from_text(text)
    train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)
    # Losses are nats per *token*, which is not comparable across tokenizers. Bits per byte of
    # raw text is: it is how many bits the model needs per byte of the held-out text.
    bits_per_byte = math.log2(math.e) * len(val_data) / len(val_text.encode("utf-8"))
    print(f"tokenizer={args.tokenizer} bytes/token={len(val_text.encode()) / len(val_data):.2f}")

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_kv_head=args.n_kv_head,
    )
    model = GPT(cfg)
    print(f"vocab={tok.vocab_size} params={model.num_params():,} train_tokens={len(train_data):,}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    t0 = time.time()
    for step in range(args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        if step % args.eval_every == 0:
            tr = estimate_loss(model, train_data, args.batch_size)
            va = estimate_loss(model, val_data, args.batch_size)
            print(
                f"step {step:5d} | train {tr:.3f} | val {va:.3f} "
                f"| val bits/byte {va * bits_per_byte:.3f} | {time.time() - t0:6.1f}s"
            )
        if step == args.steps:
            break
        x, y = get_batch(train_data, args.batch_size, args.block_size)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"config": asdict(cfg), "model": model.state_dict(), "tokenizer": tok.to_dict()}
    torch.save(ckpt, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
