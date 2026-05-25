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

The core of this repo is `src/microgpt.py`: a single-file, dependency-free GPT implementation in pure Python (attributed to @karpathy). It is deliberately written without NumPy, PyTorch, or any numeric library — every operation runs on Python scalars.

**Autograd (`Value`)** — A scalar-valued autograd engine. Each `Value` stores its data, gradient, children, and local gradients. `backward()` builds a topological sort of the computation graph and applies the chain rule in reverse.

**Model** — GPT-2 style character-level transformer. Key architectural choices:
- RMSNorm instead of LayerNorm
- ReLU instead of GeLU  
- No biases anywhere
- KV cache accumulated per forward pass (lists `keys`, `values` per layer)
- Token + positional embeddings summed, then RMSNorm applied before the first layer

**Parameters** — All trainable parameters live in `state_dict`, a dict of named matrices of `Value` objects (`wte`, `wpe`, `lm_head`, and per-layer `attn_*`/`mlp_*`). The flat `params` list is a flattened view of `state_dict` that the optimizer iterates over; the two must stay in sync. Architecture shape is determined by the hyperparameters and frozen into the matrix dimensions. ~4192 params at default config.

**Hyperparameters** (top of file): `n_layer=1`, `n_embd=16`, `block_size=16`, `n_head=4`. Training runs Adam with linear LR decay for `num_steps=1000` steps over `data/input.txt` (character-level name data).

**Inference**: temperature-controlled sampling (`temperature=0.5`). BOS token is used as both start and end-of-sequence sentinel.

## Conventions

- **Stay dependency-free.** Do not add NumPy, PyTorch, or other numeric/ML libraries to `src/microgpt.py`. Pure-Python scalar math is the entire point. Standard-library modules (`json`, `os`, etc.) are fine. Test/dev-only dependencies (e.g. `pytest`) are fine in the dev group.
- **TDD is required for non-exploratory code** (see `.skills` / project skills). Write a failing test first, watch it fail, then write minimal code to pass. This applies to persistence, tokenizer, and config plumbing — not to throwaway research notebooks.
- **A checkpoint is weights + tokenizer + config**, not weights alone. Saving only `state_dict` floats is insufficient: the tokenizer (`uchars`/`BOS`/`vocab_size`) and shape config (`n_layer`/`n_embd`/`block_size`/`n_head`) must travel with the weights or loaded models produce garbage.
- **Saved values are plain floats, not `Value` objects.** Strip to `.data` on save; re-wrap in `Value(...)` on load. Never serialize the computation graph (`_children`/`_local_grads`).
- **Importing `microgpt.py` must not trigger training.** Training/inference run under `if __name__ == "__main__":` so tests can import `save_checkpoint`/`load_checkpoint` etc. without a 1000-step training run.

## Layout

- `src/microgpt.py` — the model.
- `tests/` — pytest suite.
- `data/input.txt` — character-level name data (auto-downloaded if absent).
- `checkpoints/` — saved models (filename encodes seed + n_embd + n_layer + block_size).
- `notebooks/` — exploratory research.

## Research focus (TODO.txt)

The project studies statistical properties of LLM outputs: input vs. output character distributions, effect of temperature on output distributions, hallucination rates, and recursive feeding (feeding generated names back into the model). Planned extensions include saving model state across runs, experimenting with different seeds, and filtering training data by name length.