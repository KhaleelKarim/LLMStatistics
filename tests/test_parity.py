"""
Behavioral parity: PyTorch GPT vs. pure-Python reference forward pass.

Both implementations run with identical float64 weights on the same input.
Logits must agree to within 1e-8. If they don't, the translation has a bug.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import torch
import torch.nn.functional as F
from microgpt import GPT

# ---- Pure-Python reference implementation (no autograd, plain floats) ----

def _softmax(logits):
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def _rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def _linear(x, w):
    return [sum(wij * xj for wij, xj in zip(row, x)) for row in w]

def _ref_forward(token_id, pos_id, weights, n_head, head_dim, n_layer):
    """Pure-Python reference: same math as original microgpt, no autograd."""
    wte = weights['wte']
    wpe = weights['wpe']
    n_embd = len(wte[0])

    x = [wte[token_id][j] + wpe[pos_id][j] for j in range(n_embd)]
    x = _rmsnorm(x)

    for li in range(n_layer):
        x_res = x[:]
        x = _rmsnorm(x)
        q = _linear(x, weights[f'layer{li}_attn_wq'])
        k = _linear(x, weights[f'layer{li}_attn_wk'])
        v = _linear(x, weights[f'layer{li}_attn_wv'])
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            # single token (pos_id=0): only one key/value in the cache
            k_h0 = k[hs:hs + head_dim]
            v_h0 = v[hs:hs + head_dim]
            score = sum(q_h[j] * k_h0[j] for j in range(head_dim)) / head_dim ** 0.5
            attn_w = _softmax([score])  # [1.0] — only one position
            head_out = [attn_w[0] * v_h0[j] for j in range(head_dim)]
            x_attn.extend(head_out)
        x = _linear(x_attn, weights[f'layer{li}_attn_wo'])
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x[:]
        x = _rmsnorm(x)
        x = _linear(x, weights[f'layer{li}_mlp_fc1'])
        x = [max(0.0, xi) for xi in x]
        x = _linear(x, weights[f'layer{li}_mlp_fc2'])
        x = [a + b for a, b in zip(x, x_res)]

    return _linear(x, weights['lm_head'])


def test_parity_single_token():
    """PyTorch and pure-Python reference give identical logits for token 0, position 0."""
    vocab_size = 5
    n_embd = 4
    n_head = 2
    n_layer = 1
    block_size = 4
    head_dim = n_embd // n_head

    # Use float64 in PyTorch so both engines work in the same precision
    model = GPT(vocab_size, n_embd, block_size, n_layer, n_head).double()
    model.eval()

    # Extract weights as nested Python float lists for the reference
    ref_weights = {
        key.replace('.weight', ''): tensor.tolist()
        for key, tensor in model.state_dict().items()
    }

    with torch.no_grad():
        keys = [[] for _ in range(n_layer)]
        vals = [[] for _ in range(n_layer)]
        torch_logits = model(0, 0, keys, vals).tolist()

    ref_logits = _ref_forward(0, 0, ref_weights, n_head, head_dim, n_layer)

    assert len(torch_logits) == len(ref_logits) == vocab_size
    for i, (t, r) in enumerate(zip(torch_logits, ref_logits)):
        assert abs(t - r) < 1e-8, (
            f"logit[{i}] mismatch: torch={t:.12f}  ref={r:.12f}  diff={abs(t-r):.2e}"
        )
