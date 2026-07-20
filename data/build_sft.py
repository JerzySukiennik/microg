"""
Build the instruction-tuning dataset.

Pretraining taught MicroG what Polish looks like. It did not teach it that a
question is followed by an answer — in web text a question is usually followed
by more questions. This stage fixes that, and only that.

Two things make it different from pretraining:

  Chat template. Turns are wrapped in the control tokens reserved in the
  tokenizer before pretraining started, so no embedding resize is needed.

  Loss masking. The model is scored only on the assistant's reply. Training on
  the user's turn as well would teach it to invent questions, which is exactly
  the behaviour we are trying to remove.

Output: pl_sft_tokens.bin (uint16) and pl_sft_mask.bin (uint8), same length.
Mask 1 = a position the loss is computed on.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

SOURCES = [
    # (repo, weight of trust). The cleaned set is the backbone; the second adds
    # coverage at the cost of translation artefacts, which clean() handles.
    "saillab/alpaca-polish-cleaned",
    "Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish",
]

U, A, EOT = "<|user|>", "<|assistant|>", "<|endoftext|>"


def read_token(env=Path(".env")):
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def clean(s) -> str:
    """Undo the damage machine translation left in these corpora.

    Several rows arrive with their quoting baked into the value rather than
    around it — the field literally contains `'Oceń to zdanie'`. Left alone the
    model learns to sprinkle stray quotes through its answers.
    """
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    # strip one layer of wrapping quotes, repeatedly
    while len(s) > 1 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    s = s.replace("\\n", "\n").replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def format_example(instruction, inp, output):
    """One conversation turn, split into (prompt, reply) or None.

    The two halves are returned separately and tokenised separately. Slicing a
    single encoded string by character offset would be guessing: BPE merges can
    straddle any boundary, so the token count of a prefix is not guaranteed to
    equal the length of that prefix inside the whole. Here the split point sits
    immediately after a special token, which the tokenizer never merges across,
    so encoding the halves apart is both safe and exact.
    """
    instruction, inp, output = clean(instruction), clean(inp), clean(output)
    if len(instruction) < 4 or len(output) < 2:
        return None
    user = f"{instruction}\n{inp}" if inp else instruction
    return f"{U}\n{user}\n{A}\n", output + EOT


def build(tokenizer_path: Path, out_prefix: Path, max_len: int):
    tok = Tokenizer.from_file(str(tokenizer_path))
    for t in (U, A, EOT):
        assert tok.token_to_id(t) is not None, f"{t} missing from tokenizer"

    token_buf, mask_buf = [], []
    kept = dropped = 0
    seen = set()

    for name in SOURCES:
        ds = load_dataset(name, split="train", token=read_token())
        print(f"{name}: {len(ds):,} rows", flush=True)
        for row in ds:
            made = format_example(row.get("instruction"), row.get("input"),
                                  row.get("output"))
            if made is None:
                dropped += 1
                continue
            prompt, reply = made

            # Exact-duplicate drop: the two corpora overlap heavily, and a
            # small model will happily memorise anything it sees twice.
            key = hash(prompt + reply)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)

            p_ids = tok.encode(prompt).ids
            r_ids = tok.encode(reply).ids
            ids = p_ids + r_ids
            if len(ids) > max_len:
                dropped += 1
                continue
            # Score the reply only. Training on the user's turn would teach the
            # model to invent questions — the habit we are here to remove.
            mask = np.zeros(len(ids), dtype=np.uint8)
            mask[len(p_ids):] = 1

            token_buf.append(np.asarray(ids, dtype=np.uint16))
            mask_buf.append(mask)
            kept += 1
            if kept % 10000 == 0:
                print(f"  {kept:,} kept", flush=True)

    tokens = np.concatenate(token_buf)
    masks = np.concatenate(mask_buf)
    assert len(tokens) == len(masks)

    # Where each example begins in the concatenated stream. Training samples
    # windows from these offsets rather than from anywhere, so a window never
    # opens midway through a reply — the model would otherwise be asked to
    # continue an answer whose question it never saw.
    offsets = np.zeros(len(token_buf), dtype=np.int64)
    np.cumsum([len(a) for a in token_buf[:-1]], out=offsets[1:])

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(f"{out_prefix}_tokens.bin")
    masks.tofile(f"{out_prefix}_mask.bin")
    offsets.tofile(f"{out_prefix}_offsets.bin")

    print(f"\nkept {kept:,} examples, dropped {dropped:,}")
    print(f"{len(tokens)/1e6:.1f}M tokens, {masks.mean()*100:.1f}% trained on")
    print(f"-> {out_prefix}_tokens.bin / _mask.bin")

    print("\n--- sample (special tokens shown) ---")
    print(repr(tok.decode([int(t) for t in token_buf[0]],
                          skip_special_tokens=False))[:420])
    print(f"(loss on {int(mask_buf[0].sum())} of {mask_buf[0].size} tokens)")

    # Alignment audit across the whole set, not one lucky example: the token
    # immediately before the first scored position must be the newline that
    # follows <|assistant|>. If this drifts, the model is trained to predict
    # the wrong half of the conversation and nothing else will reveal it.
    a_id, bad = tok.token_to_id(A), 0
    for t_arr, m_arr in zip(token_buf[:5000], mask_buf[:5000]):
        first = int(np.argmax(m_arr))
        if first < 2 or int(t_arr[first - 2]) != a_id:
            bad += 1
    print(f"alignment check: {bad} of {min(5000, len(token_buf))} examples misaligned")
    assert bad == 0, "loss mask does not start right after <|assistant|>"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer-v2.json"))
    ap.add_argument("--out-prefix", type=Path, default=Path("data/pl_sft"))
    ap.add_argument("--max-len", type=int, default=768)
    args = ap.parse_args()
    build(args.tokenizer, args.out_prefix, args.max_len)
    sys.stdout.flush()
    os._exit(0)   # see fetch_corpus.py — datasets can abort at interpreter exit
