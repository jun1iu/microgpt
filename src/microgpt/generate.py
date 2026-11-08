"""Sample text from a trained checkpoint."""

from __future__ import annotations

import argparse

import torch

from microgpt.data import CharTokenizer
from microgpt.model import GPT, GPTConfig


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, CharTokenizer(ckpt["chars"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/model.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--no-cache", action="store_true", help="disable the KV cache")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, tok = load_checkpoint(args.ckpt)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    out = model.generate(
        idx,
        args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=not args.no_cache,
        generator=g,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
