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

**Hyperparameters** (top of file): `n_layer=1`, `n_embd=16`, `block_size=16`, `n_head=4`. Training runs Adam with linear LR decay for `num_steps=1000` steps over `data/input.txt` (character-level name data).

**Inference**: temperature-controlled sampling (`temperature=0.5`). BOS token is used as both start and end-of-sequence sentinel.

## Research focus (TODO.txt)

The project studies statistical properties of LLM outputs: input vs. output character distributions, effect of temperature on output distributions, hallucination rates, and recursive feeding (feeding generated names back into the model). Planned extensions include saving model state across runs, experimenting with different seeds, and filtering training data by name length.

Data lives in `data/input.txt`. Notebooks go in `notebooks/`.
