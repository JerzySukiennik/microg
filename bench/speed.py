"""
Generation throughput on the target hardware — offline, on a CPU, which is
the actual product promise (see README: "an Intel MacBook Pro is the
target"). Formalises the ad-hoc KV-cache measurements from earlier in
development into a repeatable check.

Reports both cached and uncached tok/s at a realistic context length, since
the gap between them (the whole reason KV-cache exists) is itself part of
the story.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402
sys.path.insert(0, str(REPO))
from model.gpt import GPTConfig  # noqa: E402


def _bench(model, prompt_len, n_new, use_cache, warmup=3):
    cfg = model.config
    idx = torch.randint(0, cfg.vocab_size, (1, prompt_len))
    for _ in model.generate(idx, warmup, use_cache=use_cache):
        pass
    t0 = time.time()
    n = 0
    for _ in model.generate(idx, n_new, use_cache=use_cache):
        n += 1
    return n / (time.time() - t0)


def run(model, prompt_len=512, n_new=40):
    torch.set_num_threads(8)
    return {
        "prompt_len": prompt_len,
        "cached_tok_s": _bench(model, prompt_len, n_new, use_cache=True),
        "uncached_tok_s": _bench(model, prompt_len, n_new, use_cache=False),
    }


if __name__ == "__main__":
    from common import load_model
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "../Niepotrzebne/kaggle-orchestration/output/run/best.pt"
    model, step, _ = load_model(ckpt)
    out = run(model)
    speedup = out["cached_tok_s"] / out["uncached_tok_s"]
    print(f"step {step} — context {out['prompt_len']} tokens")
    print(f"  with KV-cache:    {out['cached_tok_s']:.1f} tok/s")
    print(f"  without KV-cache: {out['uncached_tok_s']:.1f} tok/s")
    print(f"  speedup: {speedup:.2f}x")
