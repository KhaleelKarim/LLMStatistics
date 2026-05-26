import sys, os
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
