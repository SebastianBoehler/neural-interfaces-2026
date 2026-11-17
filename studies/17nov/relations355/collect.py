"""Whole alternatives only; unchanged official scorer after every fold is complete."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from model import ARMS
from run import sha


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--workspace', type=Path, required=True)
    args = a.parse_args()
    root, ws = args.root, args.workspace
    sys.path[:0] = [str(root/'experiments'), str(ws/'evaluator')]
    import protocol
    from score_s_v66p import person_accuracy, summarize
    with np.load(root/'data/proteus_windows.npz') as z:
        # Query labels remain unopened until all producer proofs and row banks pass.
        subject = z['subject'].copy()
    people = sorted(set(subject))
    banks = {}
    for family, arms in ARMS.items():
        out = ws/'output'/family
        proof = json.loads((out/'proof.json').read_text())
        assert proof['epochs'] == 40 and len(proof['folds']) == 10 and not proof['labels_scored']
        assert proof['data_sha256'] == sha(root/'data/proteus_windows.npz')
        assert proof['source'] == {p.name:sha(p) for p in (ws/'code').glob('*.py')}
        for arm in arms:
            bank = np.full((2655,3), np.nan)
            for held, rec in zip(people, proof['folds']):
                assert rec['held'] == held and rec['source_people'] == sorted(set(people)-{held})
                rows = np.flatnonzero(subject == held)
                path = out/f'{held.split("/")[-1]}_{arm}.npz'
                assert sha(path) == rec['arms'][arm]['bank_sha256']
                assert sha(out/f'{held.split("/")[-1]}_{arm}.pt') == rec['arms'][arm]['checkpoint_sha256']
                with np.load(path) as z:
                    assert np.array_equal(z['rows'], rows)
                    bank[rows] = z['probability']
            assert np.isfinite(bank).all() and np.allclose(bank.sum(1), 1, atol=1e-6)
            banks[arm] = bank
    orders = ('within0','within1','within2','within3','full0','full1','runs0','runs1')
    with np.load(root/'runs/shuffle_perms.npz') as z:
        permutations = {o:z[f'{o[:-1]}|{o[-1]}|all_loso'].copy() for o in orders}
    for order in permutations.values():
        assert np.array_equal(np.sort(order), np.arange(2655))
        for bank in banks.values():
            delivered = bank[order]
            restored = np.empty_like(delivered); restored[order] = delivered
            assert np.array_equal(restored, bank)
    (ws/'before_labels.json').write_text(json.dumps(dict(complete_families=2, folds=20,
        banks=40, order_zero_exact=True, stateless=True,
        proofs={f:sha(ws/'output'/f/'proof.json') for f in ARMS}), indent=2))
    with np.load(root/'data/proteus_windows.npz') as z:
        d = {k:z[k].copy() for k in ('subject','session','interface')}
        y = z['y'].copy()
    d['stream'] = np.char.add(np.char.add(d['subject'],'|'),d['session'])
    cell = np.array(['|'.join(v) for v in zip(d['subject'],d['session'],d['interface'])])
    person = np.array([p.split('/')[-1] for p in d['subject']])
    short = tuple(p.split('/')[-1] for p in people)
    points, pp = {}, {}
    for arm, bank in banks.items():
        points[arm] = 100*protocol.score(y, bank.argmax(1), d, np.ones(len(y),bool))['balanced_accuracy_cell_mean']
        pp[arm] = person_accuracy(bank,y,cell,person,short)
    result = dict(scope='WHOLE NEW-PERSON alternatives; no current-system/returning/sealed claim',
        official_equal_cell=points,
        absolute={a:summarize(np.tile(v,(8,1)),orders,short) for a,v in pp.items()},
        contrasts={f'{a}_minus_{b}':summarize(np.tile(pp[a]-pp[b],(8,1)),orders,short)
                   for a,b in (('lag6','instant'),('aligned','offset'))},
        note='Stateless predictions make all eight delivery orders identical; interval uncertainty is people only.')
    (ws/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
