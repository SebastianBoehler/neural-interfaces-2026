"""Check the complete G recording twice with the profiling optimizer disabled."""
import json
from pathlib import Path

import numpy as np
import torch

import replay
from stock_trace import compare, observe


def main():
    reference = next(iter(json.loads(Path('/prior/cap_5.json').read_text()).values()))
    rows = np.array(reference['rows'])
    with np.load('/external.npz') as archive:
        x = torch.from_numpy(archive['X'])
        names = archive['ch_names'].tolist()
    template = replay.load(Path('/package'), names)
    with torch.jit.optimized_execution(False):
        cold = observe(template, x, rows)
        warm = observe(template, x, rows)
    output = compare(cold, warm, reference['labels'])
    output['cold_vs_archive'] = np.flatnonzero(np.array(cold[0]) != reference['labels']).tolist()
    output['terminal_104'] = [cold[1][104]['out'], warm[1][104]['out']]
    Path('graph_endpoint.json').write_text(json.dumps(output))


if __name__ == '__main__':
    main()
