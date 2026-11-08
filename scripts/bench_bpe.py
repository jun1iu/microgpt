"""BPE tokenizer: training time, compression on held-out text, and encode throughput.

    python scripts/bench_bpe.py --data data/shakespeare.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from microgpt.bpe import BPETokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/shakespeare.txt")
    args = p.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    cut = int(0.9 * len(text))
    train_text, val_text = text[:cut], text[cut:]
    val_bytes = len(val_text.encode("utf-8"))
    print(f"{len(train_text):,} train chars, {val_bytes:,} held-out bytes\n")
    print("| vocab | train time (s) | bytes/token (held-out) | encode (MB/s) | round-trip |")
    print("|---:|---:|---:|---:|:-:|")
    for vocab in (300, 512, 1024, 2048):
        t0 = time.perf_counter()
        tok = BPETokenizer.train(train_text, vocab)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ids = tok.encode(val_text)
        enc_s = time.perf_counter() - t0
        ok = tok.decode(ids) == val_text
        print(
            f"| {vocab} | {train_s:.1f} | {val_bytes / len(ids):.2f} "
            f"| {val_bytes / enc_s / 1e6:.2f} | {'yes' if ok else 'NO'} |"
        )


if __name__ == "__main__":
    main()
