"""
Train a byte-level BPE tokenizer for Polish.

Why not reuse GPT-2's tokenizer? It was fitted to English text, so Polish
morphology shatters into fragments — "przeuczony" costs ~6 tokens instead of 2.
Every wasted token is wasted context and wasted compute, so at 117M we cannot
afford it.

Why BPE at all? We need a vocabulary that is finite (the model has one row per
token) but can still spell *any* string. BPE gets both: start from raw bytes —
256 symbols that cover everything — then repeatedly find the most frequent
adjacent pair and merge it into a new symbol. Frequent whole words end up as
single tokens; rare words fall back to pieces; nothing is ever unrepresentable.

The merge *training* runs in Rust (the `tokenizers` library) because doing it in
Python over a multi-GB corpus would take days. The algorithm is the paragraph
above — nothing is hidden.
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE

# Reserved control tokens. The chat format is built out of these, so they must
# exist in the vocabulary from the very start of pretraining — you cannot bolt
# them on later without the embeddings being untrained noise.
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|context|>",   # marks retrieved vault fragments injected by RAG
]


def train(corpus_files: list[Path], vocab_size: int, out_path: Path):
    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=True))

    # Split on whitespace/punctuation boundaries before merging, and keep the
    # leading space attached to the following word. That way "kot" and " kot"
    # are distinct tokens and the model never has to guess about spacing.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train([str(p) for p in corpus_files], trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"saved -> {out_path}  (vocab {tokenizer.get_vocab_size()})")
    return tokenizer


def report_efficiency(tokenizer: Tokenizer):
    """Sanity check: how many tokens does ordinary Polish actually cost?"""
    samples = [
        "Cześć, nazywam się Jurek i buduję rakiety w Gzowie.",
        "Przeuczony model językowy generuje niespójne odpowiedzi.",
        "Wydrukowałem stateczniki na drukarce Bambu Lab X1C.",
    ]
    print("\n--- tokenizer efficiency ---")
    for s in samples:
        ids = tokenizer.encode(s).ids
        print(f"{len(ids):3d} tokens / {len(s):3d} chars  ({len(s)/len(ids):.2f} chars per token)")
        assert tokenizer.decode(ids) == s, "roundtrip failed!"
    print("roundtrip: ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+", type=Path, help="plain-text corpus files")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--out", type=Path, default=Path("data/tokenizer.json"))
    args = ap.parse_args()

    tok = train(args.corpus, args.vocab_size, args.out)
    report_efficiency(tok)
