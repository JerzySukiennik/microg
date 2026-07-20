"""
Kaggle cell 2 of 2 — pretraining.

Settings: Accelerator GPU T4 x2, Internet ON, Persistence "Variables and Files".
Inputs: the 'microg-data' Dataset produced by 01-prep.py, plus — on any run
after the first — the previous run's output as 'microg-ckpt'.

A Kaggle session is capped at 12 hours and can die sooner without warning, so
this is built to be interrupted. Everything needed to continue lands in
/kaggle/working/run every CKPT_EVERY steps; add that output as an input to the
next session and it picks up mid-stride.
"""

import os
import shutil
import subprocess
import sys
import glob

REPO = "https://github.com/JerzySukiennik/microg.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run"

# ---------------------------------------------------------------- schedule --
# 16 x 30 x 1024 = 491,520 tokens per step. 4060 steps = 2.0B tokens, matching
# the packed corpus exactly — one epoch, no repeats.
BATCH, ACCUM, STEPS, WARMUP = 16, 30, 4060, 200

if not os.path.exists(f"{WORK}/microg"):
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/microg"], check=True)
os.chdir(f"{WORK}/microg")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"], check=True)

# ------------------------------------------------------------------- data --
data_dir = next((d for d in glob.glob("/kaggle/input/*")
                 if os.path.exists(f"{d}/pl_train.bin")), None)
assert data_dir, "no input dataset containing pl_train.bin — run 01-prep.py first"
print(f"data: {data_dir}")

# ------------------------------------------------------- resume if possible --
os.makedirs(OUT, exist_ok=True)
prev = next((p for p in glob.glob("/kaggle/input/*/run/ckpt.pt")), None)
resume = []
if prev:
    shutil.copy(prev, f"{OUT}/ckpt.pt")
    best = prev.replace("ckpt.pt", "best.pt")
    if os.path.exists(best):
        shutil.copy(best, f"{OUT}/best.pt")
    resume = ["--resume"]
    print(f"resuming from {prev}")
else:
    print("starting from scratch")

# ------------------------------------------------------------------ train --
cmd = [sys.executable, "train/train.py",
       "--data", f"{data_dir}/pl",
       "--out", OUT,
       "--batch-size", str(BATCH),
       "--grad-accum", str(ACCUM),
       "--max-steps", str(STEPS),
       "--warmup", str(WARMUP),
       "--eval-every", "100",
       "--ckpt-every", "100",     # ~13 min of work at risk if the session dies
       "--log-every", "10"] + resume
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

print("\nsave this notebook's output as a Dataset ('microg-ckpt') to continue "
      "in the next session, or download run/best.pt if training finished.")
