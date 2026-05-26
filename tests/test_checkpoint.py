import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import microgpt
from microgpt import Value, save_checkpoint, load_checkpoint

# ---------------------------------------------------------------------------
# Test 1: weights round-trip — saved floats come back with correct values
# ---------------------------------------------------------------------------

def test_weights_roundtrip(tmp_path):
    sd = {
        'a': [[Value(1.5), Value(-0.3)], [Value(0.7), Value(2.1)]],
    }
    path = str(tmp_path / "ckpt.json")
    save_checkpoint(path, sd, list('abc'), 3, n_layer=1, n_embd=2, block_size=2, n_head=1)
    loaded_sd, *_ = load_checkpoint(path)
    assert loaded_sd['a'][0][0].data == 1.5
    assert loaded_sd['a'][0][1].data == -0.3
    assert loaded_sd['a'][1][0].data == 0.7
    assert loaded_sd['a'][1][1].data == 2.1

# ---------------------------------------------------------------------------
# Test 2: loaded values are Value objects, not raw floats
# ---------------------------------------------------------------------------

def test_loaded_values_are_Value_objects(tmp_path):
    sd = {'w': [[Value(3.14)]]}
    path = str(tmp_path / "ckpt.json")
    save_checkpoint(path, sd, list('abc'), 3, n_layer=1, n_embd=1, block_size=1, n_head=1)
    loaded_sd, *_ = load_checkpoint(path)
    assert isinstance(loaded_sd['w'][0][0], Value)

# ---------------------------------------------------------------------------
# Test 3: saved file contains plain numbers, not serialized Value objects
# ---------------------------------------------------------------------------

def test_saved_file_contains_plain_numbers(tmp_path):
    sd = {'w': [[Value(1.5), Value(-0.3)]]}
    path = str(tmp_path / "ckpt.json")
    save_checkpoint(path, sd, list('ab'), 2, n_layer=1, n_embd=2, block_size=1, n_head=1)
    with open(path) as f:
        raw = json.load(f)
    for row in raw['weights']['w']:
        for v in row:
            assert isinstance(v, float), f"expected float, got {type(v)}: {v!r}"

# ---------------------------------------------------------------------------
# Test 4: params list is correctly rebuilt — right length and references the
#         same Value objects that live in the loaded state_dict (not stale copies)
# ---------------------------------------------------------------------------

def test_params_rebuilt_after_load(tmp_path):
    sd = {
        'a': [[Value(1.0), Value(2.0)], [Value(3.0), Value(4.0)]],
        'b': [[Value(5.0)]],
    }
    path = str(tmp_path / "ckpt.json")
    save_checkpoint(path, sd, list('xy'), 2, n_layer=1, n_embd=2, block_size=2, n_head=1)
    loaded_sd, loaded_params, *_ = load_checkpoint(path)

    expected_len = sum(len(row) for mat in loaded_sd.values() for row in mat)
    assert len(loaded_params) == expected_len

    # every param must be the exact object found in loaded_sd (identity, not equality)
    sd_flat = [p for mat in loaded_sd.values() for row in mat for p in row]
    for p, q in zip(loaded_params, sd_flat):
        assert p is q

# ---------------------------------------------------------------------------
# Test 5: tokenizer round-trips — uchars, BOS, vocab_size identical after load
# ---------------------------------------------------------------------------

def test_tokenizer_roundtrip(tmp_path):
    original_uchars = ['a', 'b', 'c']
    original_BOS = 3
    sd = {'w': [[Value(0.0)]]}
    path = str(tmp_path / "ckpt.json")
    save_checkpoint(path, sd, original_uchars, original_BOS, n_layer=1, n_embd=1, block_size=1, n_head=1)
    _, _, loaded_uchars, loaded_BOS, *_ = load_checkpoint(path)

    assert loaded_uchars == original_uchars
    assert loaded_BOS == original_BOS

# ---------------------------------------------------------------------------
# Test 6: config round-trips — n_layer, n_embd, block_size, n_head identical
# ---------------------------------------------------------------------------

def test_config_roundtrip(tmp_path):
    sd = {'w': [[Value(0.0)]]}
    path = str(tmp_path / "ckpt.json")
    save_checkpoint(path, sd, ['a'], 1, n_layer=3, n_embd=32, block_size=8, n_head=2)
    _, _, _, _, loaded_n_layer, loaded_n_embd, loaded_block_size, loaded_n_head = load_checkpoint(path)

    assert loaded_n_layer == 3
    assert loaded_n_embd == 32
    assert loaded_block_size == 8
    assert loaded_n_head == 2
