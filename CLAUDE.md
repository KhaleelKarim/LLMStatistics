# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for package management (Python 3.12).

```bash
# Install dependencies
uv sync

# Run training + inference
uv run python src/microgpt.py

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_foo.py::test_name -v
```

`microgpt.py` must be run from the project root — it references `data/input.txt` via a relative path.

## Architecture

The core of this repo is `src/microgpt.py`: a PyTorch GPT implementation (ported from the pure-Python original attributed to @karpathy). It uses PyTorch for compute and autograd while preserving the same architecture choices.

**Model (`GPT`)** — `nn.Module` subclass. GPT-2 style character-level transformer. Key architectural choices:
- RMSNorm instead of LayerNorm (no learnable scale — standalone `rmsnorm()` function)
- ReLU instead of GeLU
- No biases anywhere
- KV cache accumulated per forward pass (lists `keys`, `values` per layer, same interface as original)
- Token + positional embeddings summed, then RMSNorm applied before the first layer

**Parameters** — All trainable parameters are `nn.Parameter` tensors registered via `setattr` in `GPT.__init__`. Access via `model.state_dict()` or `model.parameters()`. Layer names follow the convention `layer{i}_attn_wq`, `layer{i}_mlp_fc1`, etc. ~4192 params at default config.

**Checkpoints** — Saved as `.pt` files via `torch.save` / `torch.load`. Format: `{'config': {...}, 'tokenizer': {...}, 'model': model.state_dict()}`. The `save_checkpoint` / `load_checkpoint` functions reconstruct the full `GPT` object from the saved config.

**Hyperparameters** (top of file): `n_layer=1`, `n_embd=16`, `block_size=16`, `n_head=4`. Training runs Adam with linear LR decay for `num_steps=1000` steps over `data/input.txt` (character-level name data).

**Inference**: temperature-controlled sampling (`temperature=0.5`). BOS token is used as both start and end-of-sequence sentinel.

## Conventions

- **`src/microgpt.py` uses PyTorch for compute.** Do not add other numeric/ML libraries to it. Standard-library modules (`os`, `math`, etc.) are fine. Test/dev-only dependencies (e.g. `pytest`) are fine in the dev group.
- **TDD is required for non-exploratory code** (see `.skills` / project skills). Write a failing test first, watch it fail, then write minimal code to pass. This applies to persistence, tokenizer, and config plumbing — not to throwaway research notebooks.
- **A checkpoint is weights + tokenizer + config**, not weights alone. Saving only model weights is insufficient: the tokenizer (`uchars`/`BOS`) and shape config (`n_layer`/`n_embd`/`block_size`/`n_head`) must travel with the weights or loaded models produce garbage.
- **Checkpoints are `.pt` files** written by `torch.save` and read by `torch.load`. The `save_checkpoint` / `load_checkpoint` functions in `microgpt.py` are the canonical interface.
- **Importing `microgpt.py` must not trigger training.** Training/inference run under `if __name__ == "__main__":` so tests can import `save_checkpoint`/`load_checkpoint` etc. without a 1000-step training run.

## Layout

- `src/microgpt.py` — the model.
- `tests/` — pytest suite.
- `data/input.txt` — character-level name data (auto-downloaded if absent).
- `checkpoints/` — saved models as `.pt` files (filename encodes seed + n_embd + n_layer + block_size).
- `notebooks/` — exploratory research.

## Research focus (TODO.txt)

The project studies statistical properties of LLM outputs: input vs. output character distributions, effect of temperature on output distributions, hallucination rates, and recursive feeding (feeding generated names back into the model). Planned extensions include saving model state across runs, experimenting with different seeds, and filtering training data by name length.