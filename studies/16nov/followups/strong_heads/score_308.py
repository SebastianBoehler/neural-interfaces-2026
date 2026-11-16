"""Use the original full-chain scorer, adding matched seed-averaged contrasts."""

import os
import sys
from pathlib import Path


def patch_source(original, score):
    source = score.patch_score(original)
    source = score.replace_once(source, 'from member_chain import ARMS, ORDERS', 'from chain_308 import ARMS, ORDERS')
    anchor = '    def contrast(a, b):'
    addition = '''    for arm in HEADS:
        for order in ORDERS:
            metric[arm, order] = (metric[arm + '0', order] + metric[arm + '1', order]) / 2
            official[arm, order] = (official[arm + '0', order] + official[arm + '1', order]) / 2
'''
    source = score.replace_once(source, anchor, addition + anchor)
    old = "pairs = [('Nfixed', 'control'), ('Ntrain', 'control'), ('Ntrain', 'Nfixed')]"
    pairs = "pairs = [(a, 'control') for a in HEADS] + [('Stanh','Slinear'), ('Rfourier','Rlinear'), ('Rquadratic','Rlinear')]"
    source = score.replace_once(source, old, pairs)
    old_report_arms = "for a in ('control', *ARMS[1:])"
    if source.count(old_report_arms) != 3:
        raise ValueError('unexpected native score report arm anchors')
    source = source.replace(old_report_arms, "for a in ('control', *ARMS[1:], *HEADS)")
    source = source.replace('native_core_s301_results.json', 'strong_heads_s308_results.json')
    return source


def main():
    root = Path(os.environ['NI26_WS']).resolve(strict=True)
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(root / 'studies/16nov/followups/native_core'),
                    str(root / 'studies/16nov/followups/geometry')]
    from chain_308 import ARMS, HEADS  # noqa: E402
    import score  # noqa: E402
    path = root / 'studies/16nov/followups/geometry/score.py'
    source = patch_source(path.read_text(), score)
    context = {'__name__': 's308_score', '__file__': str(path), 'HEADS': HEADS}
    exec(compile(source, str(path), 'exec'), context)
    score.ARMS = ARMS
    context['verify_geometry'] = score.verify_native
    context['main']()


if __name__ == '__main__':
    main()
