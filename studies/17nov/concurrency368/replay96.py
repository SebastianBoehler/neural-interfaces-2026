"""Exact §366 public prefix in a fresh worker process."""
import json
from pathlib import Path
import resource
import sys

import numpy as np
import torch

from replay import load, read_source, stream


def main(output):
    if torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('expected exactly one A100')
    x, _, channels = read_source()
    rows = np.random.default_rng(36017800).permutation(len(x))[:96]
    torch.cuda.reset_peak_memory_stats()
    result = stream(load(Path('/package'), channels), x, rows)
    a = result['activity']
    if a['recognized'] != -1 or not a['ft_refits'] or not a['cp_fused']:
        raise RuntimeError(f'expected active adaptation: {a}')
    result.update(rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  allocated_bytes=torch.cuda.max_memory_allocated(),
                  reserved_bytes=torch.cuda.max_memory_reserved())
    if 'neural-interfaces-2026/.venv/' in Path('/proc/self/maps').read_text():
        raise RuntimeError('worker mapped a host environment dependency')
    Path(output).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
