import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import microgpt
from microgpt import Value, save_checkpoint, load_checkpoint, build_filename, should_train, build_kl_filename

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

# ---------------------------------------------------------------------------
# Test 7: behavioral fidelity — same logits before save and after load
# ---------------------------------------------------------------------------

def test_behavioral_fidelity(tmp_path):
    path = str(tmp_path / "ckpt.json")

    # capture logits from the live model
    keys = [[] for _ in range(microgpt.n_layer)]
    vals  = [[] for _ in range(microgpt.n_layer)]
    before = [v.data for v in microgpt.gpt(0, 0, keys, vals)]

    save_checkpoint(path, microgpt.state_dict, microgpt.uchars, microgpt.BOS,
                    microgpt.n_layer, microgpt.n_embd, microgpt.block_size, microgpt.n_head)

    loaded_sd, *_ = load_checkpoint(path)
    original_sd = microgpt.state_dict
    microgpt.state_dict = loaded_sd
    try:
        keys = [[] for _ in range(microgpt.n_layer)]
        vals  = [[] for _ in range(microgpt.n_layer)]
        after = [v.data for v in microgpt.gpt(0, 0, keys, vals)]
    finally:
        microgpt.state_dict = original_sd

    assert before == after

# ---------------------------------------------------------------------------
# Test 8: build_filename encodes config and distinguishes different configs
# ---------------------------------------------------------------------------

def test_build_filename():
    name = build_filename(seed=42, n_embd=16, n_layer=1, block_size=16)
    assert isinstance(name, str)
    assert name.endswith('.json')
    assert '42' in name
    assert name.startswith('checkpoints/')

    other = build_filename(seed=42, n_embd=32, n_layer=2, block_size=8)
    assert name != other

# ---------------------------------------------------------------------------
# Test 9: should_train returns True when no checkpoint exists, False when it does
# ---------------------------------------------------------------------------

def test_should_train(tmp_path):
    path = str(tmp_path / "ckpt.json")
    assert should_train(path) is True

    sd = {'w': [[Value(1.0)]]}
    save_checkpoint(path, sd, ['a'], 1, n_layer=1, n_embd=1, block_size=1, n_head=1)
    assert should_train(path) is False

# ---------------------------------------------------------------------------
# Test 10: loading a checkpoint whose config mismatches expected shapes raises
# ---------------------------------------------------------------------------

def test_load_config_mismatch_raises(tmp_path):
    import pytest
    # save a checkpoint with n_embd=4
    sd = {'wte': [[Value(0.1)] * 4] * 2}
    path = str(tmp_path / "ckpt.json")
    save_checkpoint(path, sd, ['a', 'b'], 2, n_layer=1, n_embd=4, block_size=2, n_head=1)

    # tamper: overwrite config to claim n_embd=8 while weights are still shape 4
    with open(path) as f:
        payload = json.load(f)
    payload['config']['n_embd'] = 8
    with open(path, 'w') as f:
        json.dump(payload, f)

    with pytest.raises((AssertionError, ValueError)):
        load_checkpoint(path)

# ---------------------------------------------------------------------------
# Test 11: build_kl_filename encodes config and kl_interval
# ---------------------------------------------------------------------------

def test_build_kl_filename():
    name = build_kl_filename(seed=42, n_embd=16, n_layer=1, block_size=16, kl_interval=10)
    assert name.startswith('data/')
    assert name.endswith('.npz')
    assert '42' in name and '10' in name

    other = build_kl_filename(seed=42, n_embd=16, n_layer=1, block_size=16, kl_interval=5)
    assert name != other
