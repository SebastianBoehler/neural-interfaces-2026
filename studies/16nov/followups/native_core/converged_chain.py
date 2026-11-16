"""Reuse the native full-chain wrapper for the four registered §305 arms."""
import os
import sys
from pathlib import Path

ARMS = ('Rzero', 'Nfixed', 'Rconverged', 'Nconverged')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def main():
    path = Path(__file__).with_name('member_chain.py')
    source = path.read_text()
    old = "ARMS = ('Rzero', 'Nfixed', 'Ntrain')"
    if source.count(old) != 1:
        raise ValueError('unexpected native member arm declaration')
    source = source.replace(old, f'ARMS = {ARMS!r}', 1)
    anchor = '            self.track(Path(__file__).resolve())'
    if source.count(anchor) != 1:
        raise ValueError('unexpected native member tracker')
    source = source.replace(anchor, anchor + f'\n            self.track({str(Path(__file__).resolve())!r})', 1)
    exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path), 'os': os, 'sys': sys})


if __name__ == '__main__':
    main()
