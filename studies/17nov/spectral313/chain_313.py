"""Reuse the exact native-core full-chain wrapper for five spectral components."""

import os
import sys
from pathlib import Path

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
ARMS = ('Rzero', 'BASE', 'POWER', 'REAL', 'IMAG', 'BOTH')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def main():
    path = ROOT / 'studies/16nov/followups/native_core/member_chain.py'
    source = path.read_text()
    old = "ARMS = ('Rzero', 'Nfixed', 'Ntrain')"
    if source.count(old) != 1:
        raise ValueError('native chain arm anchor changed')
    source = source.replace(old, f'ARMS = {ARMS!r}', 1)
    anchor = '            self.track(Path(__file__).resolve())'
    if source.count(anchor) != 1:
        raise ValueError('native chain tracker anchor changed')
    source = source.replace(anchor, anchor + f'\n            self.track({str(Path(__file__).resolve())!r})', 1)
    exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path), 'os': os, 'sys': sys})


if __name__ == '__main__':
    main()
