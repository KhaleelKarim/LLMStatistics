import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pytest
from microgpt import GPT, save_checkpoint, load_checkpoint, build_filename, should_train, build_kl_filename


def _tiny_gpt(n_embd=4, n_head=2, n_layer=1, block_size=4, vocab_size=5):
    return GPT(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_layer=n_layer, n_head=n_head)


# ---------------------------------------------------------------------------
# Test 1: weights round-trip — saved tensors come back with correct values
# ---------------------------------------------------------------------------

def test_weights_roundtrip(tmp_path):
    model = _tiny_gpt()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, list('abcd'), 4, n_layer=1, n_embd=4, block_size=4, n_head=2)
    loaded_model, *_ = load_checkpoint(path)
    for key in model.state_dict():
        assert torch.allclose(model.state_dict()[key], loaded_model.state_dict()[key])

# ---------------------------------------------------------------------------
# Test 2: loaded weights are tensors, not raw floats
# ---------------------------------------------------------------------------

def test_loaded_weights_are_tensors(tmp_path):
    model = _tiny_gpt()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, list('abcd'), 4, n_layer=1, n_embd=4, block_size=4, n_head=2)
    loaded_model, *_ = load_checkpoint(path)
    for key, val in loaded_model.state_dict().items():
        assert isinstance(val, torch.Tensor), f"{key}: expected Tensor, got {type(val)}"

# ---------------------------------------------------------------------------
# Test 3: checkpoint file is a valid torch archive with expected top-level keys
# ---------------------------------------------------------------------------

def test_checkpoint_file_loads_cleanly(tmp_path):
    model = _tiny_gpt()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, list('abcd'), 4, n_layer=1, n_embd=4, block_size=4, n_head=2)
    payload = torch.load(path, weights_only=False)
    assert 'config' in payload
    assert 'tokenizer' in payload
    assert 'model' in payload

# ---------------------------------------------------------------------------
# Test 4: param count is preserved after save + load
# ---------------------------------------------------------------------------

def test_params_count_after_load(tmp_path):
    model = _tiny_gpt()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, list('abcd'), 4, n_layer=1, n_embd=4, block_size=4, n_head=2)
    loaded_model, *_ = load_checkpoint(path)
    assert (sum(p.numel() for p in loaded_model.parameters()) ==
            sum(p.numel() for p in model.parameters()))

# ---------------------------------------------------------------------------
# Test 5: tokenizer round-trips — uchars and BOS identical after load
# ---------------------------------------------------------------------------

def test_tokenizer_roundtrip(tmp_path):
    uchars = ['a', 'b', 'c']
    BOS = 3
    model = _tiny_gpt(vocab_size=len(uchars) + 1)  # must match len(uchars)+1
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, uchars, BOS, n_layer=1, n_embd=4, block_size=4, n_head=2)
    _, loaded_uchars, loaded_BOS, *_ = load_checkpoint(path)
    assert loaded_uchars == uchars
    assert loaded_BOS == BOS

# ---------------------------------------------------------------------------
# Test 6: config round-trips — n_layer, n_embd, block_size, n_head identical
# ---------------------------------------------------------------------------

def test_config_roundtrip(tmp_path):
    model = GPT(vocab_size=5, n_embd=8, block_size=8, n_layer=2, n_head=4)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, list('abcd'), 4, n_layer=2, n_embd=8, block_size=8, n_head=4)
    _, _, _, loaded_n_layer, loaded_n_embd, loaded_block_size, loaded_n_head = load_checkpoint(path)
    assert loaded_n_layer == 2
    assert loaded_n_embd == 8
    assert loaded_block_size == 8
    assert loaded_n_head == 4

# ---------------------------------------------------------------------------
# Test 7: behavioral fidelity — same logits before save and after load
# ---------------------------------------------------------------------------

def test_behavioral_fidelity(tmp_path):
    model = _tiny_gpt()
    model.eval()
    path = str(tmp_path / "ckpt.pt")

    with torch.no_grad():
        keys = [[] for _ in range(1)]
        vals = [[] for _ in range(1)]
        before = model(0, 0, keys, vals).tolist()

    save_checkpoint(path, model, list('abcd'), 4, n_layer=1, n_embd=4, block_size=4, n_head=2)
    loaded_model, *_ = load_checkpoint(path)
    loaded_model.eval()

    with torch.no_grad():
        keys = [[] for _ in range(1)]
        vals = [[] for _ in range(1)]
        after = loaded_model(0, 0, keys, vals).tolist()

    assert before == after

# ---------------------------------------------------------------------------
# Test 8: build_filename encodes config and distinguishes different configs
# ---------------------------------------------------------------------------

def test_build_filename():
    name = build_filename(seed=42, n_embd=16, n_layer=1, block_size=16, step=1000)
    assert isinstance(name, str)
    assert name.endswith('.pt')
    assert '42' in name
    assert 'step1000' in name
    assert name.startswith('checkpoints/')

    other = build_filename(seed=42, n_embd=32, n_layer=2, block_size=8, step=1000)
    assert name != other

    different_step = build_filename(seed=42, n_embd=16, n_layer=1, block_size=16, step=500)
    assert name != different_step

# ---------------------------------------------------------------------------
# Test 9: should_train returns True when no checkpoint exists, False when it does
# ---------------------------------------------------------------------------

def test_should_train(tmp_path):
    path = str(tmp_path / "ckpt.pt")
    assert should_train(path) is True

    model = _tiny_gpt()
    save_checkpoint(path, model, list('a'), 1, n_layer=1, n_embd=4, block_size=4, n_head=2)
    assert should_train(path) is False

# ---------------------------------------------------------------------------
# Test 10: loading a checkpoint whose config mismatches the weights raises
# ---------------------------------------------------------------------------

def test_load_config_mismatch_raises(tmp_path):
    model = _tiny_gpt(n_embd=4)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, list('abcd'), 4, n_layer=1, n_embd=4, block_size=4, n_head=2)

    # tamper: claim n_embd=8 while the actual weights are shape 4
    payload = torch.load(path, weights_only=False)
    payload['config']['n_embd'] = 8
    torch.save(payload, path)

    with pytest.raises((RuntimeError, ValueError)):
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
