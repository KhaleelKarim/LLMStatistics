import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ngrams import load_docs, build_tokenizer, compute_ngram_distribution, save_ngrams


def build_output_ngrams_path(infer_path):
    stem = os.path.basename(infer_path)
    stem = os.path.splitext(stem)[0]
    stem = stem.removeprefix("infer_")
    return os.path.join("data", f"output_ngrams_{stem}.npz")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("infer_path", help="path to inference output .txt file")
    args = parser.parse_args()

    out_path = build_output_ngrams_path(args.infer_path)
    if os.path.exists(out_path):
        print(f"Found {out_path} — delete it to recompute.")
        sys.exit(0)

    ref_docs = load_docs("data/input.txt")
    uchars, BOS, vocab_size, stoi = build_tokenizer(ref_docs)

    infer_docs = load_docs(args.infer_path)
    print(f"infer docs: {len(infer_docs)}, vocab_size: {vocab_size}")

    distributions = {}
    for order in range(1, 5):
        probs, seen_mask = compute_ngram_distribution(infer_docs, stoi, BOS, vocab_size, order)
        distributions[order] = (probs, seen_mask)
        if seen_mask is not None:
            n_seen, n_total = int(seen_mask.sum()), seen_mask.size
        else:
            n_seen = n_total = vocab_size
        print(f"  order={order}: {n_seen}/{n_total} contexts seen, probs shape {probs.shape}")

    save_ngrams(out_path, distributions)
    print(f"Saved → {out_path}")
