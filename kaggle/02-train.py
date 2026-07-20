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
# 8 x 60 x 1024 = 491,520 tokens per step — same total as 16x30, deliberately:
# the DataParallel test scattered batch=16 as 8+8 across two GPUs, so one GPU
# alone OOM'd trying to hold all 16 (13.58 GiB used, 1.95 GiB more requested,
# only 1.01 GiB free). Halving batch and doubling accum keeps effective batch
# size and tokens/step identical, so the single-GPU throughput comparison
# stays apples-to-apples. 4060 steps = 2.0B tokens, matching the packed
# corpus exactly — one epoch, no repeats.
BATCH, ACCUM, STEPS, WARMUP = 8, 60, 4060, 200

if os.path.exists(f"{WORK}/microg"):
    # A stale checkout from an earlier attempt in this same session would
    # silently run old code even after this script itself was re-fetched —
    # that is exactly what ran a T4 session at half speed on the bf16 fix.
    # Pulling forces the checkout to match what curl just downloaded.
    subprocess.run(["git", "-C", f"{WORK}/microg", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/microg"], check=True)
os.chdir(f"{WORK}/microg")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"], check=True)

# ------------------------------------------------------------------- data --
# Kaggle mounts a dataset at /kaggle/input/<slug>/, but when the dataset was
# built from notebook output the files can sit one level deeper. Search both,
# and if nothing turns up, print the tree rather than just asserting — "not
# found" is useless when you cannot see what *is* there.
hits = (glob.glob("/kaggle/input/*/pl_train.bin")
        + glob.glob("/kaggle/input/*/*/pl_train.bin")
        + glob.glob("/kaggle/input/*/*/*/pl_train.bin"))
if not hits:
    print("pl_train.bin not found. /kaggle/input contains:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 3:
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files)[:12]:
            size = os.path.getsize(os.path.join(root, f)) / 1e9
            print("  " * (depth + 1) + f"{f}  {size:.2f} GB")
    raise SystemExit("attach the microg-data dataset, or wait for it to finish building")

data_dir = os.path.dirname(hits[0])
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
       "--log-every", "10",
       # TEMPORARY: measuring whether DataParallel's per-microstep model
       # broadcast over PCIe (T4 has no NVLink) is costing more than the
       # second GPU is worth. A live run held steady at ~25k tok/s with 2
       # GPUs regardless of a vectorize+prefetch fix that ruled out data
       # loading as the cause — DataParallel's replicate/gather tax is the
       # next suspect. Remove this line once the comparison is done.
       "--single-gpu"] + resume
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

print("\nsave this notebook's output as a Dataset ('microg-ckpt') to continue "
      "in the next session, or download run/best.pt if training finished.")
