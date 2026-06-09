import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from microgpt import build_infer_filename
from ngrams import load_docs


def test_build_infer_filename():
    assert build_infer_filename(42, 16, 1, 16, 100, 1000) == \
        "data/infer_seed42_embd16_layer1_blk16_n100_step1000.txt"


def test_build_infer_filename_varies_with_num_infer():
    a = build_infer_filename(42, 16, 1, 16, 20, 1000)
    b = build_infer_filename(42, 16, 1, 16, 500, 1000)
    assert a != b
    assert "n20" in a
    assert "n500" in b


def test_build_infer_filename_varies_with_step():
    a = build_infer_filename(42, 16, 1, 16, 20, 100)
    b = build_infer_filename(42, 16, 1, 16, 20, 1000)
    assert a != b
    assert "step100" in a
    assert "step1000" in b


def test_infer_file_readable_by_load_docs(tmp_path):
    names = ["emma", "olivia", "ava"]
    p = tmp_path / "infer.txt"
    p.write_text('\n'.join(names) + '\n')
    assert load_docs(str(p)) == names
