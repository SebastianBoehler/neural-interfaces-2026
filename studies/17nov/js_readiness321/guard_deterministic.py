"""Strict phase-matched guard under the validated controlled CUDA executor only."""
import os
import sys
from pathlib import Path

if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('workspace config must precede CUDA')
import torch
import phase_guard


def main(variant):
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    phase_guard.CONTROL=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_deterministic1/ft_repeat.json')
    phase_guard.main(variant)


if __name__=='__main__':main(sys.argv[1])
