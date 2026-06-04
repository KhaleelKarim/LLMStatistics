"""
GPT in PyTorch — fast, GPU-capable rewrite.
Architecture: RMSNorm (no learnable scale), ReLU activations, no biases, KV cache.
Identical architectural choices to the pure-Python original; computation replaces Value scalars with tensors.
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 42
random.seed(seed)
torch.manual_seed(seed)

if not os.path.exists('data/input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'data/input.txt')
docs = [line.strip() for line in open('data/input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        for i in range(n_layer):
            setattr(self, f'layer{i}_attn_wq', nn.Linear(n_embd, n_embd, bias=False))
            setattr(self, f'layer{i}_attn_wk', nn.Linear(n_embd, n_embd, bias=False))
            setattr(self, f'layer{i}_attn_wv', nn.Linear(n_embd, n_embd, bias=False))
            setattr(self, f'layer{i}_attn_wo', nn.Linear(n_embd, n_embd, bias=False))
            setattr(self, f'layer{i}_mlp_fc1', nn.Linear(n_embd, 4 * n_embd, bias=False))
            setattr(self, f'layer{i}_mlp_fc2', nn.Linear(4 * n_embd, n_embd, bias=False))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.08)

    def forward(self, token_id: int, pos_id: int, keys: list, values: list) -> torch.Tensor:
        device = self.wte.weight.device
        x = (self.wte(torch.tensor([token_id], device=device)) +
             self.wpe(torch.tensor([pos_id], device=device))).squeeze(0)
        x = rmsnorm(x)
        for li in range(self.n_layer):
            x_res = x
            x = rmsnorm(x)
            q = getattr(self, f'layer{li}_attn_wq')(x)
            k = getattr(self, f'layer{li}_attn_wk')(x)
            v = getattr(self, f'layer{li}_attn_wv')(x)
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs + self.head_dim]
                k_h = torch.stack([ki[hs:hs + self.head_dim] for ki in keys[li]])
                v_h = torch.stack([vi[hs:hs + self.head_dim] for vi in values[li]])
                attn_w = F.softmax(k_h @ q_h / self.head_dim ** 0.5, dim=0)
                x_attn.append(attn_w @ v_h)
            x = getattr(self, f'layer{li}_attn_wo')(torch.cat(x_attn))
            x = x + x_res
            x_res = x
            x = rmsnorm(x)
            x = F.relu(getattr(self, f'layer{li}_mlp_fc1')(x))
            x = getattr(self, f'layer{li}_mlp_fc2')(x)
            x = x + x_res
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def build_filename(seed, n_embd, n_layer, block_size):
    return f"checkpoints/ckpt_seed{seed}_embd{n_embd}_layer{n_layer}_blk{block_size}.pt"

def build_kl_filename(seed, n_embd, n_layer, block_size, kl_interval):
    return f"data/kl_seed{seed}_embd{n_embd}_layer{n_layer}_blk{block_size}_interval{kl_interval}.npz"

def build_infer_filename(seed, n_embd, n_layer, block_size, num_infer):
    return f"data/infer_seed{seed}_embd{n_embd}_layer{n_layer}_blk{block_size}_n{num_infer}.txt"

def should_train(path):
    return not os.path.exists(path)

def save_checkpoint(path, model, uchars, BOS, n_layer, n_embd, block_size, n_head):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'config': {'n_layer': n_layer, 'n_embd': n_embd, 'block_size': block_size, 'n_head': n_head},
        'tokenizer': {'uchars': uchars, 'BOS': BOS},
        'model': model.state_dict(),
    }, path)

def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload['config']
    tok = payload['tokenizer']
    n_layer_c, n_embd_c = cfg['n_layer'], cfg['n_embd']
    block_size_c, n_head_c = cfg['block_size'], cfg['n_head']
    uchars_c = tok['uchars']
    BOS_c = tok['BOS']
    vocab_size_c = len(uchars_c) + 1
    model = GPT(vocab_size_c, n_embd_c, block_size_c, n_layer_c, n_head_c)
    model.load_state_dict(payload['model'])
    model.to(device)
    return model, uchars_c, BOS_c, n_layer_c, n_embd_c, block_size_c, n_head_c


# ---------------------------------------------------------------------------
# Entry point: load checkpoint if available, otherwise train then save
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from ngrams import load_ngrams
    from kl import kl_at_position, average_kl, save_kl_records

    parser = argparse.ArgumentParser()
    parser.add_argument('--kl', type=int, default=10, metavar='INTERVAL',
                        help='compute KL divergence every INTERVAL steps (0=off, default=10)')
    parser.add_argument('--num_infer', type=int, default=20, metavar='N',
                        help='number of names to generate during inference (default: 20)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='compute device: cpu, cuda, mps (default: cpu)')
    args = parser.parse_args()
    kl_interval = args.kl
    device = args.device

    NGRAMS_PATH = "data/ngrams.npz"
    kl_path = build_kl_filename(seed, n_embd, n_layer, block_size, kl_interval)

    distributions = None
    if kl_interval > 0:
        if os.path.exists(NGRAMS_PATH):
            distributions = load_ngrams(NGRAMS_PATH)
        else:
            print(f"KL tracking disabled: {NGRAMS_PATH} not found. Run: uv run python src/ngrams.py")
            kl_interval = 0

    ckpt_path = build_filename(seed, n_embd, n_layer, block_size)

    if should_train(ckpt_path):
        model = GPT(vocab_size, n_embd, block_size, n_layer, n_head).to(device)
        print(f"num params: {sum(p.numel() for p in model.parameters())}")

        learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     betas=(beta1, beta2), eps=eps_adam)

        num_steps = 1000
        kl_records = {1: [], 2: [], 3: [], 4: []}

        for step in range(num_steps):
            doc = docs[step % len(docs)]
            tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
            n = min(block_size, len(tokens) - 1)

            is_eval = kl_interval > 0 and step % kl_interval == 0
            if is_eval:
                step_kl_positions = []

            keys_c = [[] for _ in range(n_layer)]
            vals_c = [[] for _ in range(n_layer)]
            losses = []

            for pos_id in range(n):
                token_id_t, target_id = tokens[pos_id], tokens[pos_id + 1]
                logits = model(token_id_t, pos_id, keys_c, vals_c)
                probs = F.softmax(logits, dim=-1)
                losses.append(-probs[target_id].log())
                if is_eval:
                    step_kl_positions.append(kl_at_position(
                        probs.detach().cpu().numpy(), tokens, pos_id, distributions
                    ))

            loss = sum(losses) / n

            optimizer.zero_grad()
            loss.backward()

            lr_t = learning_rate * (1 - step / num_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_t
            optimizer.step()

            if is_eval:
                step_avg = average_kl(step_kl_positions)
                for order, val in step_avg.items():
                    if not math.isnan(val):
                        kl_records[order].append((step, val))

            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

        save_checkpoint(ckpt_path, model, uchars, BOS, n_layer, n_embd, block_size, n_head)
        print(f"\ncheckpoint saved → {ckpt_path}")
        if kl_interval > 0:
            save_kl_records(kl_path, kl_records)
            print(f"KL records saved → {kl_path}")
    else:
        print(f"loading checkpoint from {ckpt_path}")
        if kl_interval > 0:
            print("Note: KL tracking only runs during training. Delete checkpoint to re-enable.")
        model, uchars, BOS, n_layer, n_embd, block_size, n_head = load_checkpoint(ckpt_path, device)
        head_dim = n_embd // n_head
        vocab_size = len(uchars) + 1

    # Inference: may the model babble back to us
    num_infer = args.num_infer
    infer_path = build_infer_filename(seed, n_embd, n_layer, block_size, num_infer)
    temperature = 0.5
    print("\n--- inference (new, hallucinated names) ---")
    generated = []
    model.eval()
    with torch.no_grad():
        for sample_idx in range(num_infer):
            keys_c = [[] for _ in range(n_layer)]
            vals_c = [[] for _ in range(n_layer)]
            token_id = BOS
            sample = []
            for pos_id in range(block_size):
                logits = model(token_id, pos_id, keys_c, vals_c)
                probs = F.softmax(logits / temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                if token_id == BOS:
                    break
                sample.append(uchars[token_id])
            name = ''.join(sample)
            generated.append(name)
            print(f"sample {sample_idx+1:2d}: {name}")

    os.makedirs("data", exist_ok=True)
    with open(infer_path, 'w') as f:
        f.write('\n'.join(generated) + '\n')
    print(f"inference output saved → {infer_path}")
