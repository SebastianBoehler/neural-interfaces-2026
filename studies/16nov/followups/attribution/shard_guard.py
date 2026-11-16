"""Validate one source-locked §300 shard, including its exact input hashes."""
import json
from pathlib import Path
import numpy as np


def load(root, order, arms, bases, score, probability, smoke=False):
    folder = 's300_shards_smoke' if smoke else 's300_shards'
    path = Path(root) / f'runs/{folder}/{order}.npz'; side = path.with_suffix('.json')
    manifest = json.loads(side.read_text())
    expected = {f'{a}|{stage}|{b}' for a in arms for stage in ('V6Fr', 'count', 'CP') for b in bases}
    if manifest.get('order') != order or manifest.get('smoke') is not smoke or manifest.get('sha256') != score.sha256(path):
        raise ValueError(f'{order}: shard identity or hash mismatch')
    if set(manifest.get('keys', [])) != expected: raise ValueError(f'{order}: incomplete shard manifest')
    score.verify_inputs(manifest.get('input_sha256'), {}, root)
    result = {}
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != expected: raise ValueError(f'{order}: incomplete shard arrays')
        for key in expected:
            a = z[key].copy()
            if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all(): raise ValueError(f'{key}: invalid array')
            if '|CP|' not in key: probability(a)
            result[key] = a
    return result
