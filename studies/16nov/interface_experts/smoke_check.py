"""Check real-data smoke without reading held labels or scores."""
import json
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import guards
import inputs
from style import sha


def main():
    root = Path(os.environ['EXPERT_ROOT']).resolve(strict=True)
    lock = root / 'runs/input306/inputs.json'
    if sha(lock) != lock.with_suffix('.sha256').read_text().split()[0]: raise ValueError('bad lock')
    with np.load(inputs.PROD / 'data/proteus_windows.npz', allow_pickle=False) as z:
        subject, session = z['subject'].copy(), z['session'].copy()
    with np.load(inputs.PROD / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in guards.ORDERS}
    meta, _ = guards.load_fold(root / 'runs/expert306_smoke/P01.npz', 'P01', subject, session, perms, sha(lock), 'smoke')
    marker = root / 'runs/expert306_smoke/passed.json'
    with marker.open('x') as f:
        json.dump(dict(input_lock_sha256=sha(lock), smoke_sha256=sha(marker.parent / 'P01.npz'),
                       seconds=meta['seconds'], original_max_error=meta['original_max_error']), f, sort_keys=True)
        f.write('\n')
    print('real-data expert smoke passed; no held accuracy', flush=True)


if __name__ == '__main__': main()
