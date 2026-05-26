import os
import math
import numpy as np


def kl_divergence(p, q, eps=1e-8):
    q_s = (q + eps) / (q + eps).sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q_s[mask])))


def kl_at_position(p_model_values, tokens, pos_id, distributions, orders=(1, 2, 3, 4), eps=1e-8):
    p = np.array([v.data for v in p_model_values])
    result = {}
    for order in orders:
        if pos_id < order - 2:
            result[order] = None
            continue
        probs_n, seen_n = distributions[order]
        if order == 1:
            q = probs_n
        else:
            ctx = tuple(tokens[pos_id - order + 2 : pos_id + 1])
            if not seen_n[ctx]:
                result[order] = None
                continue
            q = probs_n[ctx]
        result[order] = kl_divergence(p, q, eps)
    return result


def average_kl(per_position_results):
    if not per_position_results:
        return {}
    all_orders = list(per_position_results[0].keys())
    accum = {o: [] for o in all_orders}
    for d in per_position_results:
        for order, val in d.items():
            if val is not None:
                accum[order].append(val)
    return {o: float(np.mean(vals)) if vals else float('nan') for o, vals in accum.items()}


def save_kl_records(path, kl_records):
    arrays = {}
    for order, pairs in kl_records.items():
        if pairs:
            steps, kl_vals = zip(*pairs)
        else:
            steps, kl_vals = [], []
        arrays[f'steps_{order}'] = np.array(steps, dtype=np.int64)
        arrays[f'kl_{order}'] = np.array(kl_vals, dtype=np.float64)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_kl_records(path):
    data = np.load(path)
    records = {}
    for key in data.files:
        if key.startswith('steps_'):
            order = int(key.split('_')[1])
            records[order] = (data[f'steps_{order}'], data[f'kl_{order}'])
    return records
