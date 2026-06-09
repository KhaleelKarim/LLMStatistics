"""Run microgpt.py with seed=75 / 32000 steps / 32000 samples per snapshot.

Produces 33 inference files (steps 0, 1000, ..., 32000) plus one final checkpoint.
Idempotent: if the step-32000 checkpoint already exists, the script exits without re-running.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

# Stale if n_embd/n_layer/block_size in microgpt.py change.
FINAL_CKPT = REPO_ROOT / "checkpoints" / "ckpt_seed75_embd16_layer1_blk16_step32000.pt"
if FINAL_CKPT.exists():
    print(f"already done: {FINAL_CKPT.name} exists. Delete to re-run.")
    sys.exit(0)

cmd = [
    "uv", "run", "python", "src/microgpt.py",
    "--num_infer", "32000",
    "--seed", "75",
    "--num_steps", "32000",
    "--train_interval", "1000",
]
print("running:", " ".join(cmd))
sys.exit(subprocess.run(cmd).returncode)
