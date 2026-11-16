"""Use the original complete full-chain scorer with registered spectral contrasts."""

import os
import sys
from pathlib import Path


def patch_source(original, score):
    source = score.patch_score(original)
    source = score.replace_once(source, 'from member_chain import ARMS, ORDERS', 'from chain_313 import ARMS, ORDERS')
    old = "pairs = [('Nfixed', 'control'), ('Ntrain', 'control'), ('Ntrain', 'Nfixed')]"
    pairs = "pairs = [(a, 'control') for a in ARMS[1:]] + [('POWER','BASE'), ('REAL','POWER'), ('IMAG','REAL'), ('BOTH','REAL'), ('BOTH','BASE')]"
    source = score.replace_once(source, old, pairs)
    return source.replace('native_core_s301_results.json', 'spectral_s313_chain_results.json')


def main():
    root = Path(os.environ['NI26_WS']).resolve(strict=True)
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(root / 'studies/16nov/followups/native_core'),
                    str(root / 'studies/16nov/followups/geometry')]
    from chain_313 import ARMS  # noqa: E402
    import score  # noqa: E402
    path = root / 'studies/16nov/followups/geometry/score.py'
    context = {'__name__': 's313_chain_score', '__file__': str(path)}
    exec(compile(patch_source(path.read_text(), score), str(path), 'exec'), context)
    score.ARMS = ARMS
    context['verify_geometry'] = score.verify_native
    context['main']()


if __name__ == '__main__':
    main()
