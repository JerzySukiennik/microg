"""
Perplexity on held-out Polish text — the headline "how good is this base
model" number, and the one directly comparable to GPT-2-class reference
points.

Reuses pl_val.bin: text the packer routed to validation and the model never
trained on (see data/pack_data.py — split is by whole document, not by
token, so no sentence leaks across the train/val boundary).

Samples fixed-seed windows so repeated runs across checkpoints are
comparable — a different random sample each time would make "did it improve"
partly a measurement artefact.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402


def run(model, block_size=1024, n_windows=200, seed=1234):
    data = np.memmap(REPO / "data" / "pl_val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - block_size - 1, size=n_windows)

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for s in starts:
            chunk = torch.from_numpy(data[s:s + block_size + 1].astype(np.int64))
            x, y = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
            _, loss = model(x, targets=y, return_logits=False)
            total_loss += loss.item() * (block_size)
            total_tokens += block_size

    mean_loss = total_loss / total_tokens
    import math
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss),
            "n_windows": n_windows, "block_size": block_size, "tokens": total_tokens}


if __name__ == "__main__":
    from common import load_model
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "../Niepotrzebne/kaggle-orchestration/output/run/best.pt"
    model, step, _ = load_model(ckpt)
    out = run(model)
    print(f"step {step} — loss {out['loss']:.4f}  perplexity {out['perplexity']:.2f}  "
          f"({out['n_windows']} windows x {out['block_size']} tokens)")
