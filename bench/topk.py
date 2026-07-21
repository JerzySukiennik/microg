"""
Top-k next-token accuracy — the intuitive counterpart to perplexity.

Perplexity answers "how surprised was the model, on average" in a log-scale
that is hard to feel. This answers a plainer question: at each position in
held-out text, is the actual next word among the model's top 1 / top 5
guesses? Same held-out windows as perplexity.py, same fixed seed, so the two
benchmarks are drawn from literally the same text.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402


def run(model, block_size=1024, n_windows=200, seed=1234, ks=(1, 5)):
    data = np.memmap(REPO / "data" / "pl_val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - block_size - 1, size=n_windows)

    hits = {k: 0 for k in ks}
    total = 0
    max_k = max(ks)

    with torch.no_grad():
        for s in starts:
            chunk = torch.from_numpy(data[s:s + block_size + 1].astype(np.int64))
            x, y = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
            logits, _ = model(x, targets=y, return_logits=True)  # (1, T, V)
            topk_ids = logits.topk(max_k, dim=-1).indices[0]      # (T, max_k)
            target = y[0].unsqueeze(-1)                            # (T, 1)
            for k in ks:
                hits[k] += (topk_ids[:, :k] == target).any(dim=-1).sum().item()
            total += y.numel()

    return {f"top{k}_accuracy": hits[k] / total for k in ks} | {"tokens": total}


if __name__ == "__main__":
    from common import load_model
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "../Niepotrzebne/kaggle-orchestration/output/run/best.pt"
    model, step, _ = load_model(ckpt)
    out = run(model)
    parts = [f"{name} {v*100:.1f}%" for name, v in out.items() if name != "tokens"]
    print(f"step {step} — " + "  ".join(parts))
