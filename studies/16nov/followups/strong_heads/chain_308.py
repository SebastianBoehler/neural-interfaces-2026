"""Reuse the verified native-member chain with §308's explicit seed arms."""

import os
import sys
from pathlib import Path

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from definitions import ARMS as HEADS, ORDERS, SEED_ARMS as ARMS  # noqa: E402


def main():
    path = ROOT / 'studies/16nov/followups/native_core/member_chain.py'
    source = path.read_text()
    old = "ARMS = ('Rzero', 'Nfixed', 'Ntrain')"
    if source.count(old) != 1:
        raise ValueError('native chain arm anchor changed')
    source = source.replace(old, f'ARMS = {ARMS!r}', 1)
    anchor = '            self.track(Path(__file__).resolve())'
    if source.count(anchor) != 1:
        raise ValueError('native chain source tracker changed')
    source = source.replace(anchor, anchor + f'\n            self.track({str(Path(__file__).resolve())!r})', 1)
    exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path), 'os': os, 'sys': sys})


if __name__ == '__main__':
    main()
