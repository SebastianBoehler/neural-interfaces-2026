"""Use the guarded full-chain scorer for all four registered §305 arms."""
import os
import sys
from pathlib import Path


def main():
    root = Path(os.environ['NI26_WS']).resolve(strict=True)
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(root / 'studies/16nov/followups/geometry')]
    import score  # noqa: E402
    from converged_chain import ARMS  # noqa: E402
    path = root / 'studies/16nov/followups/geometry/score.py'
    source = score.patch_score(path.read_text())
    old = 'from member_chain import ARMS, ORDERS'
    if source.count(old) != 1:
        raise ValueError('unexpected native scorer arm import')
    source = source.replace(old, 'from converged_chain import ARMS, ORDERS', 1)
    old = "pairs = [('Nfixed', 'control'), ('Ntrain', 'control'), ('Ntrain', 'Nfixed')]"
    if source.count(old) != 1:
        raise ValueError('unexpected native scorer contrasts')
    source = source.replace(old, "pairs = [('Nfixed', 'control'), ('Rconverged', 'control'), ('Nconverged', 'control'), ('Nconverged', 'Nfixed'), ('Rconverged', 'Nconverged')]", 1)
    source = source.replace('native_core_s301_results.json', 'native_core_s305_results.json')
    context = {'__name__': 's305_score', '__file__': str(path)}
    exec(compile(source, str(path), 'exec'), context)
    score.ARMS = ARMS
    context['verify_geometry'] = score.verify_native
    context['main']()


if __name__ == '__main__':
    main()
