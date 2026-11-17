"""One fresh worker process with identical warm-up and workload for each arm."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from replay import load, read_source, stream


def main(phase, arm, mode, replicate):
    start = perf_counter()
    if torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('expected one A100')
    if phase == 'smoke':
        pieces = [torch.arange(3, device='cuda', dtype=d) / 3 for d in (torch.float32, torch.float64)]
        if not torch.equal(torch.stack(pieces).cpu(), torch.stack([p.cpu() for p in pieces])):
            raise ValueError('device stack promotion changes posterior')
        pieces[1][0] = torch.nan
        if torch.isfinite(torch.stack(pieces).cpu()).all().item():
            raise ValueError('nonfinite posterior escaped check')
    x, sessions, channels = read_source()
    if mode == 'ordinary':
        key, rows = sessions[0]
        n = 48 if phase == 'smoke' else 96
    elif mode == 'stress':
        key = 'forced-unknown-378'
        rows = np.random.default_rng(36017800).permutation(len(x))
        n = 96
    else:
        raise ValueError(mode)
    template = load(Path('/package'), channels)
    warm = stream(template, x, rows[:32])
    result = stream(template, x, rows[:n])
    reference = json.loads(Path(f'/work/{mode}.json').read_text())['streams']['first'][key]['labels']
    if warm['labels'] != reference[:32] or result['labels'] != reference[:n]:
        raise ValueError(f'archived label mismatch: {mode} {arm}')
    result['wall_s'] = perf_counter() - start
    result['warm_activity'] = warm['activity']
    if mode == 'stress' and (result['activity']['ft_steps'] != 160 or result['activity']['cp_fused'] != 6):
        raise ValueError('active work differs')
    Path(f'/work/{phase}-{mode}-{replicate}-{arm}.json').write_text(json.dumps(result) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
