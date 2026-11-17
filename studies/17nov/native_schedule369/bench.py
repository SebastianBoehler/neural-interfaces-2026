"""Same process and pinned A100 for three native CP configurations."""
import json
from pathlib import Path

import numpy as np
import torch

from replay import load, read_source, stream
from variants import STAGES


def main():
    if torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('expected exactly one A100')
    x, _, channels = read_source()
    rows = np.random.default_rng(36017800).permutation(len(x))[:378]
    stock = load(Path('/stock'), channels)
    warmup = stream(stock, x, rows[:96])
    if warmup['activity']['cp_fused'] != 6:
        raise RuntimeError('warmup did not reach active personalization')
    results = {}
    for name in ('stock', 'phase90', 'skip', 'stock_end'):
        model = stock if name.startswith('stock') else load(Path('/work')/name, channels)
        ran = stream(model, x, rows)
        results[name] = ran
        print(name, ran['s'], ran['activity'], flush=True)
    original = results['stock']
    expected = json.loads(Path('qualified_reference.json').read_text())
    anchor = expected['streams']['first']['forced-unknown-800']['labels'][:len(rows)]
    if original['labels'] != anchor:
        raise ValueError('warm stock prefix differs from qualified reference')
    if any(results[name]['labels'] != original['labels'] for name in ('stock_end', 'skip')):
        raise ValueError('skip or closing stock replay changed a stock prediction')
    for name, start in dict(STAGES, stock_end=20).items():
        baseline = dict(original['activity'], cp_refits=len(range(start, len(rows), 20)))
        if results[name]['activity'] != baseline or baseline['cp_fused'] != len(rows)-90:
            raise ValueError(f'{name} applied different work')
    results['phase90_changed'] = sum(a != b for a,b in
                                    zip(results['phase90']['labels'], original['labels'], strict=True))
    Path('bench.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
