"""Atomically record sequential looks and their exact checked output hashes."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/lustre/work/gehler/gfh527/ni26_rolling_20261116')


def paths_for(root, arms, orders, bases):
    paths = []
    for arm in arms:
        for order in orders:
            paths.extend(root / f'runs/newmember/s281_v66p_{arm}_{order}_{kind}'
                         for kind in ('gate.npz', 'provenance.json'))
            paths.append(root / f'runs/chainout/s281_v66p_{arm}_{order}.npz')
            for base in bases:
                cp = root / f'runs/s281_v66_cp/{arm}_{order}_{base}.npz'
                paths.extend((cp, cp.with_suffix('.json')))
    return paths


def record(wave, arm, paths, prereg, digest, logroot=ROOT):
    unique = sorted({Path(p).resolve(strict=True) for p in [*paths, prereg]})
    hashes = {str(p): digest(p) for p in unique}
    logroot.mkdir(parents=True, exist_ok=True)
    path = logroot / 'reads.jsonl'
    with path.open('a+', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0); previous = [json.loads(line) for line in f if line.strip()]
        entry = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'wave': wave, 'arm': arm,
                 'already_read_blocks': [f"{x['wave']}:{x['arm']}" for x in previous],
                 'input_sha256': hashes}
        encoded = json.dumps(entry, sort_keys=True, separators=(',', ':'))
        f.seek(0, 2); f.write(encoded + '\n'); f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return path, hashlib.sha256(encoded.encode()).hexdigest(), entry['already_read_blocks']
