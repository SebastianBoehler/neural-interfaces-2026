"""Pair each trained decoder against the preceding real-only experiment on identical folds."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import score379 as reference


def main(replicate=False):
    roots = {'clean': Path(os.environ['WAVE379']), 'coherent': Path('coherent'),
             'independent': Path('independent')}
    if replicate:
        roots['clean'] = Path('clean')
        roots['original'] = Path(os.environ['WAVE379'])
    people, orders = reference.PEOPLE, reference.ORDERS
    file = reference.BASE/'data/proteus_windows.npz'
    assert reference.sha(file) == reference.DATA_SHA
    with np.load(file) as z:
        meta = {k:z[k] for k in ('subject','session','interface')}
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'), meta['session'])
    person = np.array([s.split('/')[-1] for s in meta['subject']])
    frozen, _ = reference.controls_and_age(meta)
    banks = dict(frozen)
    commitments = {}
    for kind, folder in roots.items():
        manifest = reference.source_manifest(folder/'source.sha256')
        if folder == Path(os.environ['WAVE379']):
            assert manifest == 'a2fb59c104a29bf248ec843645c9fd4adbdbd0f3ee66b36e37ca2421374119c2'
        arms, proofs = reference.folds(folder, manifest, person)
        banks.update({kind+'_'+arm: pred for arm,pred in arms.items()})
        if replicate and kind != 'original':
            for control in ('zero_write', 'source_write'):
                bank = np.full_like(frozen['frozen'], -1)
                for p in people:
                    dest = folder / (p+'.controls.npz')
                    proof = json.loads((folder/(p+'.controls.json')).read_text())
                    assert reference.sha(dest) == proof['sha']
                    assert reference.sha(folder/(p+'_memory.pt')) == proof['checkpoint']
                    with np.load(dest) as z:
                        assert np.array_equal(z['index'], np.flatnonzero(person==p))
                        bank[:,person==p] = z[control]
                    proofs[p][control] = proof['sha']
                assert np.isin(bank,(0,1,2)).all()
                banks[kind+'_'+control] = bank
        commitments[kind] = proofs
    Path('commitments.json').write_text(json.dumps(commitments,indent=2))
    with np.load(file) as z:
        labels = z['y']
    sys.path[:0] = [str(reference.METRICS),str(reference.BASE/'studies/14nov/newmember')]
    from measure import official
    from score_s_v66p import summarize, cell_weighted_summary
    assert reference.sha(reference.METRICS/'source.sha256') == '9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd'
    reference.source_manifest(reference.METRICS/'source.sha256')
    assert reference.sha(reference.BASE/'studies/14nov/newmember/score_s_v66p.py') == 'ae387cd5a7b116fc1259001d4fe449ba1065b0cb5943c117d4f65ec87a0b40d4'
    cells = np.char.add(np.char.add(meta['stream'],'|'),meta['interface'])
    counts = np.array([len(set(cells[person==p])) for p in people])
    metric = official()
    tables = {arm:np.array([[100*metric(labels[person==p], row[person==p],meta,person==p)
                            ['balanced_accuracy_cell_mean'] for p in people] for row in bank])
              for arm,bank in banks.items()}
    scores = {a:float(np.average((t[:4].mean(0)+t[4:6].mean(0)+t[6:].mean(0))/3,weights=counts))
              for a,t in tables.items()}
    original = json.loads((Path(os.environ['WAVE379'])/'result379.json').read_text())['all']['scores']
    for a,v in original.items():
        key = a if a in frozen else ('original_' if replicate else 'clean_')+a
        assert abs(scores[key]-v) < 1e-8
    def contrast(delta):
        return dict(official=cell_weighted_summary(delta,orders,counts),equal_people=summarize(delta,orders,people))
    differences = {}
    for arm in ('memory','blank','donor','norm','static'):
        for other in ('clean','independent'):
            differences[f'coherent-{other}_{arm}'] = contrast(tables['coherent_'+arm]-tables[other+'_'+arm])
    memory_gain = tables['coherent_memory'] - tables['clean_memory']
    for arm in ('blank','static','donor'):
        differences['excess_vs_'+arm] = contrast(memory_gain-(tables['coherent_'+arm]-tables['clean_'+arm]))
    if replicate:
        for arm in ('zero_write','source_write','donor','blank','static','norm'):
            differences['coherent_memory_vs_'+arm] = contrast(tables['coherent_memory']-tables['coherent_'+arm])
    result = dict(scores=scores,contrasts=differences,commitments=reference.sha(Path('commitments.json')))
    Path('result.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
