import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from output_ngrams import build_output_ngrams_path


def test_build_output_ngrams_path_basic():
    result = build_output_ngrams_path("data/infer_seed42_embd16_layer1_blk16_n3200.txt")
    assert result == "data/output_ngrams_seed42_embd16_layer1_blk16_n3200.npz"


def test_build_output_ngrams_path_different_params():
    result = build_output_ngrams_path("data/infer_seed7_embd32_layer2_blk32_n500.txt")
    assert result == "data/output_ngrams_seed7_embd32_layer2_blk32_n500.npz"
