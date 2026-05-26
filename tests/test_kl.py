import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import numpy as np
from ngrams import build_tokenizer, compute_ngram_distribution
from kl import kl_divergence, kl_at_position, average_kl, save_kl_records, load_kl_records


class FakeValue:
    def __init__(self, data): self.data = data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fake_uniform(vocab_size):
    return [FakeValue(0.0)] * vocab_size  # all-zero logits → uniform after softmax


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------

def test_kl_divergence_identical():
    p = np.array([0.3, 0.5, 0.2])
    assert abs(kl_divergence(p, p)) < 1e-6


def test_kl_divergence_known_value():
    # KL(p=[1,0] || q=[0.5,0.5]) = log(1/0.5) = log(2)
    p = np.array([1.0, 0.0])
    q = np.array([0.5, 0.5])
    assert abs(kl_divergence(p, q) - math.log(2)) < 1e-5


def test_kl_divergence_nonneg():
    p = np.array([0.4, 0.6])
    q = np.array([0.7, 0.3])
    assert kl_divergence(p, q) >= 0.0


def test_kl_divergence_handles_zero_in_q():
    # q has a zero entry — should return finite result via Laplace smoothing
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    result = kl_divergence(p, q)
    assert math.isfinite(result)
    assert result > 0


# ---------------------------------------------------------------------------
# kl_at_position
# ---------------------------------------------------------------------------

def _small_setup():
    docs = ["ab", "ba"]
    uchars, BOS, vocab_size, stoi = build_tokenizer(docs)
    distributions = {o: compute_ngram_distribution(docs, stoi, BOS, vocab_size, o)
                     for o in range(1, 5)}
    tokens = [BOS, stoi['a'], stoi['b'], BOS]
    return BOS, vocab_size, stoi, distributions, tokens


def test_kl_at_position_order1_always_valid():
    BOS, vocab_size, stoi, distributions, tokens = _small_setup()
    result = kl_at_position(fake_uniform(vocab_size), tokens, 0, distributions, orders=(1,))
    assert result[1] is not None
    assert isinstance(result[1], float)


def test_kl_at_position_skips_short_context():
    BOS, vocab_size, stoi, distributions, tokens = _small_setup()
    # order=3 requires pos_id >= 1; at pos_id=0 → None
    r0 = kl_at_position(fake_uniform(vocab_size), tokens, 0, distributions, orders=(3,))
    assert r0[3] is None
    # at pos_id=1 → (BOS, a) is a seen trigram context → float
    r1 = kl_at_position(fake_uniform(vocab_size), tokens, 1, distributions, orders=(3,))
    assert r1[3] is not None
    assert isinstance(r1[3], float)


def test_kl_at_position_skips_unseen_context():
    # Vocab from ["ab"], but bigram counts only from ["a"] → 'b' never a context
    uchars, BOS, vocab_size, stoi = build_tokenizer(["ab"])
    distributions = {2: compute_ngram_distribution(["a"], stoi, BOS, vocab_size, 2)}
    b_id = stoi['b']
    tokens = [BOS, b_id, stoi['a'], BOS]
    # pos_id=1: context = (b_id,) → unseen → None
    result = kl_at_position(fake_uniform(vocab_size), tokens, 1, distributions, orders=(2,))
    assert result[2] is None


def test_kl_at_position_known_value():
    # p_model = exact unigram probs → KL(p_unigram || p_unigram) ≈ 0
    BOS, vocab_size, stoi, distributions, tokens = _small_setup()
    probs_1, _ = distributions[1]
    p_vals = [FakeValue(float(probs_1[i])) for i in range(vocab_size)]
    result = kl_at_position(p_vals, tokens, 0, distributions, orders=(1,))
    assert result[1] is not None
    assert abs(result[1]) < 1e-5


# ---------------------------------------------------------------------------
# average_kl
# ---------------------------------------------------------------------------

def test_average_kl_ignores_none():
    results = [
        {1: 0.5, 2: 1.0, 3: None},
        {1: 0.3, 2: None, 3: 2.0},
        {1: 0.7, 2: 0.8, 3: None},
    ]
    avg = average_kl(results)
    assert abs(avg[1] - (0.5 + 0.3 + 0.7) / 3) < 1e-9
    assert abs(avg[2] - (1.0 + 0.8) / 2) < 1e-9
    assert abs(avg[3] - 2.0) < 1e-9


def test_average_kl_all_none():
    results = [{1: None, 2: None}]
    avg = average_kl(results)
    assert math.isnan(avg[1])
    assert math.isnan(avg[2])


# ---------------------------------------------------------------------------
# save_kl_records / load_kl_records
# ---------------------------------------------------------------------------

def test_save_load_kl_records_roundtrip(tmp_path):
    kl_records = {
        1: [(0, 0.5), (10, 0.4), (20, 0.3)],
        2: [(0, 1.2), (10, 1.0)],
        3: [],
    }
    path = str(tmp_path / "kl.npz")
    save_kl_records(path, kl_records)
    loaded = load_kl_records(path)

    assert set(loaded.keys()) == {1, 2, 3}

    steps_1, kl_1 = loaded[1]
    assert list(steps_1) == [0, 10, 20]
    assert np.allclose(kl_1, [0.5, 0.4, 0.3])

    steps_3, kl_3 = loaded[3]
    assert len(steps_3) == 0
    assert len(kl_3) == 0
