"""Turns GloVe word vectors into the small dataset the demo's Type tab searches.

Reads the first rows of a safetensors file plus its vocab.json, keeps the most frequent plain
words, reduces them with PCA, normalises and rounds to int8. Writes words.txt and words.fvecs.

  python3 tools/prep_words.py data/glove out_dir [--words 20000] [--dim 64] [--check]
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_head(path: Path) -> np.ndarray:
    """Rows of the single float32 matrix that are fully present in a truncated safetensors file."""
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    name = next(k for k in header if k != "__metadata__")
    info = header[name]
    assert info["dtype"] == "F32", info["dtype"]
    dim = info["shape"][1]
    start = 8 + header_len + info["data_offsets"][0]
    rows = (len(raw) - start) // (dim * 4)
    return np.frombuffer(raw, dtype="<f4", count=rows * dim, offset=start).reshape(rows, dim)


def pick_words(vocab: dict[str, int], rows: int, limit: int) -> list[tuple[str, int]]:
    """Most frequent lowercase a-z words; GloVe ids are already in frequency order."""
    keep = [
        (word, idx)
        for word, idx in vocab.items()
        if idx < rows and word.isascii() and word.isalpha() and (len(word) > 1 or word in "ai")
    ]
    keep.sort(key=lambda pair: pair[1])
    return keep[:limit]


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def reduce(full: np.ndarray, dim: int) -> np.ndarray:
    """PCA to dim components, then unit length so L2 order equals cosine order."""
    centred = full - full.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return unit(centred @ vt[:dim].T)


def quantise(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * (127.0 / np.abs(x).max())), -127, 127).astype(np.int8)


def top_k(x: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(queries @ x.T), axis=1)[:, :k]


def check(full: np.ndarray, words: list[str]) -> None:
    """How much of the full-size top 10 survives each reduction, plus a few spot checks."""
    ref = unit(full - full.mean(axis=0))
    sample = np.arange(0, len(words), max(1, len(words) // 500))
    truth = top_k(ref, ref[sample], 11)[:, 1:]
    for dim in (32, 48, 64, 96):
        q = quantise(reduce(full, dim)).astype(np.float32)
        found = top_k(q, q[sample], 11)[:, 1:]
        overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(truth, found)])
        print(f"dim {dim}: {overlap:.3f} of the 300-d top 10 kept")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("glove_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--words", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    matrix = read_head(args.glove_dir / "model.head.safetensors")
    vocab = json.loads((args.glove_dir / "vocab.json").read_text(encoding="utf-8"))["tokens"]
    picked = pick_words(vocab, len(matrix), args.words)
    words = [w for w, _ in picked]
    full = matrix[[i for _, i in picked]].astype(np.float64)
    print(f"{len(matrix)} rows read, {len(words)} words kept, last is '{words[-1]}'")

    if args.check:
        check(full, words)

    small = quantise(reduce(full, args.dim))
    index = {w: i for i, w in enumerate(words)}
    floats = small.astype(np.float32)
    for query in ("king", "ocean", "doctor"):
        near = top_k(floats, floats[[index[query]]], 8)[0][1:]
        print(f"{query}: {' '.join(words[i] for i in near)}")
    steer = unit(floats[[index["king"]]]) - unit(floats[[index["man"]]]) + unit(floats[[index["woman"]]])
    print("king - man + woman:", " ".join(words[i] for i in top_k(floats, steer, 5)[0]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "words.txt").write_text("\n".join(words) + "\n", encoding="ascii")
    with open(args.out_dir / "words.fvecs", "wb") as out:
        head = struct.pack("<i", args.dim)
        for row in floats:
            out.write(head)
            out.write(row.astype("<f4").tobytes())
    print(f"wrote {args.out_dir}/words.txt and words.fvecs ({len(words)} x {args.dim})")


if __name__ == "__main__":
    main()
