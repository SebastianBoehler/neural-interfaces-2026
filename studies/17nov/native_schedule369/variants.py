"""Create experiment-only copies with one native artifact parameter changed."""
import os
from pathlib import Path
import shutil
import sys

import torch

STAGES = {'stock': 20, 'skip': 80, 'phase90': 90}
KEY = 'v6_cp_t0'


def prepare(source, output):
    weights = torch.load(source/'native.pt', map_location='cpu', weights_only=True)
    if tuple(weights[k].item() for k in (KEY, 'v6_cp_k', 'v6_cp_w0')) != (20, 20, 90):
        raise ValueError('stock native CP configuration changed')
    for name, start in STAGES.items():
        if name == 'stock':
            continue
        target = output/name
        shutil.copytree(source, target, copy_function=os.link,
                        ignore=lambda _, names: set(names) & {'native.pt', '__pycache__'})
        value = dict(weights, **{KEY: torch.full_like(weights[KEY], start)})
        torch.save(value, target/'native.pt')
        saved = torch.load(target/'native.pt', map_location='cpu', weights_only=True)
        if saved.keys() != weights.keys() or saved[KEY].item() != start or any(
                not torch.equal(saved[k].flatten().contiguous().view(torch.uint8),
                                weights[k].flatten().contiguous().view(torch.uint8))
                for k in saved if k != KEY):
            raise ValueError(f'unexpected {name} artifact change')


if __name__ == '__main__':
    prepare(*map(Path, sys.argv[1:]))
