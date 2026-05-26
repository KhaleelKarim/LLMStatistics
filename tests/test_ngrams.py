import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ngrams import build_tokenizer, tokenize, count_ngrams, normalize_ngrams, \
    compute_ngram_distribution, save_ngrams, load_ngrams

# ---------------------------------------------------------------------------
# Test 1: unigram probabilities sum to 1
# ---------------------------------------------------------------------------

def test_unigram_sums_to_one():
    _, BOS, vocab_size, stoi = build_tokenizer(["ab"])
    counts = count_ngrams(["ab"], stoi, BOS, vocab_size, order=1)
    probs, mask = normalize_ngrams(counts)
    assert mask is None
    assert abs(probs.sum() - 1.0) < 1e-12

# ---------------------------------------------------------------------------
# Test 2: BOS token has nonzero probability in unigram
# ---------------------------------------------------------------------------

def test_unigram_bos_counted():
    _, BOS, vocab_size, stoi = build_tokenizer(["a"])
    counts = count_ngrams(["a"], stoi, BOS, vocab_size, order=1)
    assert counts[BOS] > 0

# ---------------------------------------------------------------------------
# Test 3: unigram counts targets only — BOS appears once per doc, not twice
# ---------------------------------------------------------------------------

def test_unigram_targets_only():
    # "a" → tokens [BOS, a, BOS]; targets are [a, BOS], so BOS count == 1
    _, BOS, vocab_size, stoi = build_tokenizer(["a"])
    counts = count_ngrams(["a"], stoi, BOS, vocab_size, order=1)
    assert counts[BOS] == 1

# ---------------------------------------------------------------------------
# Test 4: all seen-context rows of bigram sum to 1
# ---------------------------------------------------------------------------

def test_bigram_seen_rows_sum_to_one():
    _, BOS, vocab_size, stoi = build_tokenizer(["ab", "ba"])
    counts = count_ngrams(["ab", "ba"], stoi, BOS, vocab_size, order=2)
    probs, seen_mask = normalize_ngrams(counts)
    row_sums = probs.sum(axis=-1)
    assert np.allclose(row_sums[seen_mask], 1.0)

# ---------------------------------------------------------------------------
# Test 5: no bigrams cross document boundaries
# ---------------------------------------------------------------------------

def test_no_cross_document_bigrams():
    # "a" ends with (a→BOS); "b" starts with (BOS→b).
    # The cross-doc pair (a→b) must never appear.
    _, BOS, vocab_size, stoi = build_tokenizer(["a", "b"])
    counts = count_ngrams(["a", "b"], stoi, BOS, vocab_size, order=2)
    a_id, b_id = stoi['a'], stoi['b']
    assert counts[a_id, b_id] == 0

# ---------------------------------------------------------------------------
# Test 6: unseen bigram context gets zero row and False in seen_mask
# ---------------------------------------------------------------------------

def test_unseen_context_zero_row():
    # build vocab from "ab" so 'b' is in vocab, but count only "a"
    # → 'b' never appears as a bigram context
    _, BOS, vocab_size, stoi = build_tokenizer(["ab"])
    counts = count_ngrams(["a"], stoi, BOS, vocab_size, order=2)
    probs, seen_mask = normalize_ngrams(counts)
    b_id = stoi['b']
    assert not seen_mask[b_id]
    assert probs[b_id].sum() == 0.0

# ---------------------------------------------------------------------------
# Test 7: spot-check exact bigram counts for "aa"
# ---------------------------------------------------------------------------

def test_specific_count_value():
    # "aa" → tokens [BOS, a, a, BOS]
    # bigrams: (BOS→a), (a→a), (a→BOS)
    _, BOS, vocab_size, stoi = build_tokenizer(["aa"])
    a_id = stoi['a']
    counts = count_ngrams(["aa"], stoi, BOS, vocab_size, order=2)
    assert counts[BOS, a_id] == 1
    assert counts[a_id, a_id] == 1
    assert counts[a_id, BOS] == 1

# ---------------------------------------------------------------------------
# Test 8: block_size truncation limits the number of counted tokens
# ---------------------------------------------------------------------------

def test_block_size_truncation():
    # "abcde" → tokens length 7, n_eff without truncation = 6
    # with block_size=3, n_eff = 3, so only 3 unigram counts
    _, BOS, vocab_size, stoi = build_tokenizer(["abcde"])
    counts_trunc = count_ngrams(["abcde"], stoi, BOS, vocab_size, order=1, block_size=3)
    counts_full  = count_ngrams(["abcde"], stoi, BOS, vocab_size, order=1)
    assert counts_trunc.sum() == 3
    assert counts_full.sum() == 6


# ---------------------------------------------------------------------------
# Helpers shared by save/load tests
# ---------------------------------------------------------------------------

CORPUS = ["ab", "ba"]


def _make_distributions(orders):
    _, BOS, vocab_size, stoi = build_tokenizer(CORPUS)
    return {o: compute_ngram_distribution(CORPUS, stoi, BOS, vocab_size, o) for o in orders}


# ---------------------------------------------------------------------------
# Test 9: unigram saves and loads back correctly
# ---------------------------------------------------------------------------

def test_save_load_unigram_roundtrip(tmp_path):
    dist = _make_distributions([1])
    path = str(tmp_path / "ngrams.npz")
    save_ngrams(path, dist)
    loaded = load_ngrams(path)
    assert 1 in loaded
    probs_out, seen_out = loaded[1]
    assert seen_out is None
    assert np.allclose(dist[1][0], probs_out)

# ---------------------------------------------------------------------------
# Test 10: bigram probs and seen_mask survive roundtrip
# ---------------------------------------------------------------------------

def test_save_load_bigram_roundtrip(tmp_path):
    dist = _make_distributions([2])
    path = str(tmp_path / "ngrams.npz")
    save_ngrams(path, dist)
    loaded = load_ngrams(path)
    probs_in, mask_in = dist[2]
    probs_out, mask_out = loaded[2]
    assert np.allclose(probs_in, probs_out)
    assert np.array_equal(mask_in, mask_out)
    assert mask_out.dtype == bool

# ---------------------------------------------------------------------------
# Test 11: all four orders present after roundtrip
# ---------------------------------------------------------------------------

def test_save_load_all_orders(tmp_path):
    dist = _make_distributions([1, 2, 3, 4])
    path = str(tmp_path / "ngrams.npz")
    save_ngrams(path, dist)
    loaded = load_ngrams(path)
    assert set(loaded.keys()) == {1, 2, 3, 4}
    for order in [1, 2, 3, 4]:
        probs_in, mask_in = dist[order]
        probs_out, mask_out = loaded[order]
        assert np.allclose(probs_in, probs_out)
        if mask_in is None:
            assert mask_out is None
        else:
            assert np.array_equal(mask_in, mask_out)

# ---------------------------------------------------------------------------
# Test 12: seen_mask dtype is preserved as bool (not int8/uint8)
# ---------------------------------------------------------------------------

def test_load_preserves_seen_mask_dtype(tmp_path):
    dist = _make_distributions([2, 3])
    path = str(tmp_path / "ngrams.npz")
    save_ngrams(path, dist)
    loaded = load_ngrams(path)
    assert loaded[2][1].dtype == bool
    assert loaded[3][1].dtype == bool
