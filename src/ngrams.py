import os
import numpy as np


def load_docs(path="data/input.txt"):
    return [line.strip() for line in open(path) if line.strip()]


def build_tokenizer(docs):
    uchars = sorted(set(''.join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    stoi = {ch: i for i, ch in enumerate(uchars)}
    return uchars, BOS, vocab_size, stoi


def tokenize(doc, stoi, BOS):
    return [BOS] + [stoi[ch] for ch in doc] + [BOS]


def count_ngrams(docs, stoi, BOS, vocab_size, order, block_size=None):
    counts = np.zeros((vocab_size,) * order, dtype=np.int64)
    for doc in docs:
        tokens = tokenize(doc, stoi, BOS)
        n_eff = min(block_size, len(tokens) - 1) if block_size is not None else len(tokens) - 1
        # j is the target position; context is tokens[j-order+1 : j]
        start = max(1, order - 1)
        for j in range(start, n_eff + 1):
            window = tuple(tokens[j - order + 1 : j + 1])
            counts[window] += 1
    return counts


def normalize_ngrams(counts):
    if counts.ndim == 1:
        return counts / counts.sum(), None
    totals = counts.sum(axis=-1)
    seen_mask = totals > 0
    probs = np.zeros(counts.shape, dtype=float)
    probs[seen_mask] = counts[seen_mask] / totals[seen_mask][:, np.newaxis]
    return probs, seen_mask


def compute_ngram_distribution(docs, stoi, BOS, vocab_size, order, block_size=None):
    return normalize_ngrams(count_ngrams(docs, stoi, BOS, vocab_size, order, block_size))


def save_ngrams(path, distributions):
    arrays = {}
    for order, (probs, seen_mask) in distributions.items():
        arrays[f'probs_{order}'] = probs
        if seen_mask is not None:
            arrays[f'seen_{order}'] = seen_mask
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    np.savez_compressed(path, **arrays)


def load_ngrams(path):
    data = np.load(path)
    distributions = {}
    for key in data.files:
        if key.startswith('probs_'):
            order = int(key.split('_')[1])
            seen_key = f'seen_{order}'
            seen_mask = data[seen_key] if seen_key in data.files else None
            distributions[order] = (data[key], seen_mask)
    return distributions


if __name__ == "__main__":
    OUT_PATH = "data/ngrams.npz"
    if os.path.exists(OUT_PATH):
        print(f"Found {OUT_PATH} — delete it to recompute.")
    else:
        docs = load_docs()
        uchars, BOS, vocab_size, stoi = build_tokenizer(docs)
        print(f"docs: {len(docs)}, vocab_size: {vocab_size}")
        distributions = {}
        for order in range(1, 5):
            probs, seen_mask = compute_ngram_distribution(docs, stoi, BOS, vocab_size, order)
            distributions[order] = (probs, seen_mask)
            if seen_mask is not None:
                n_seen, n_total = int(seen_mask.sum()), seen_mask.size
            else:
                n_seen = n_total = vocab_size
            print(f"  order={order}: {n_seen}/{n_total} contexts seen, probs shape {probs.shape}")
        save_ngrams(OUT_PATH, distributions)
        print(f"Saved → {OUT_PATH}")
