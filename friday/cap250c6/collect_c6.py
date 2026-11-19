"""Score locked causal predictions, then compute the separate gold-assisted bound."""
import json
from pathlib import Path

import numpy as np
import score379 as ref
from collect import interval
from measure import official
from produce import public_inputs

from produce_c6 import ARMS, LINEAGE

THRESHOLD = 100*(.845-.8197668759)-1.110
ROLES = {'new':ref.PEOPLE,'returning':('P04','P10')}
PAIRS = (('causal','stock'),('causal','fixed'),('fixed','stock'),
         ('old','stock'),('oracle','stock'))


def read_banks():
    source = ref.sha(Path('source.sha256'))
    banks, proofs = {}, {}
    for role in ROLES:
        for order in ref.ORDERS:
            path = Path(f'{role}_{order}.npz')
            record = json.loads(path.with_suffix('.json').read_text())
            assert record['role'] == role and record['order'] == order and record['source'] == source
            assert record['output'] == ref.sha(path)
            with np.load(path,allow_pickle=False) as z:
                banks[role,order] = dict(z)
            proofs[path.stem] = record
    Path('before_labels.json').write_text(json.dumps(proofs,indent=2)+'\n')
    return banks


def collect():
    banks = read_banks()
    meta,_ = public_inputs()
    with np.load(ref.BASE/'data/proteus_windows.npz',allow_pickle=False) as z:
        labels = z['y']
    metric = official()
    result = {}
    for role,people in ROLES.items():
        table = {a: np.zeros((len(people),8)) for a in (*ARMS,'oracle')}
        exposure = np.zeros(len(people))
        changes = {a+'-'+b: np.zeros(3,int) for a,b in PAIRS}
        scope = np.ones(len(labels),bool) if role == 'new' else (
            np.isin(meta['person'],people) & (meta['session'].astype(str)!='1'))
        for j,order in enumerate(ref.ORDERS):
            b = banks[role,order]
            rows = b['rows']
            assert np.array_equal(np.sort(rows),np.flatnonzero(scope))
            y = labels[rows]
            b['oracle'] = np.where(b['old']==y,b['old'],b['stock'])
            for a,c in PAIRS:
                first, second = b[a] == y, b[c] == y
                changes[a+'-'+c] += [np.count_nonzero(b[a]!=b[c]),
                                    np.count_nonzero(first&~second),
                                    np.count_nonzero(second&~first)]
            for i,p in enumerate(people):
                mask = scope & (meta['person']==p)
                for arm in table:
                    # Scores use canonical recording indices, including the person-grouped returning archive.
                    pred = np.full(len(labels),-1,dtype=int)
                    pred[rows] = b[arm]
                    m = metric(labels[mask],pred[mask],meta,mask)
                    table[arm][i,j] = 100*m['balanced_accuracy_cell_mean']
                    exposure[i] = m['n_cells']
        assert exposure.sum() == (28 if role=='new' else 8)
        weights = exposure if role=='new' else np.array([.2,.8])
        absolute = {a:interval(t,weights)['pp'] for a,t in table.items()}
        assert abs(absolute['stock']-(76.10391516713594 if role=='new' else 86.57350585321704)) < 1e-5
        expected = json.loads(LINEAGE.read_text())['results']
        reference = (expected['all_loso']['official']['old'] if role=='new'
                     else expected['seen_session']['assumed_returning']['old'])
        assert abs(absolute['old']-reference) < 1e-8
        effects = {}
        for a,b in PAIRS:
            key = a+'-'+b
            contrast = interval(table[a]-table[b],weights)
            contrast['people'] = dict(zip(people,contrast['people']))
            effects[key] = dict(**contrast,**dict(zip(
                ('changed','corrections','regressions'),changes[key].tolist())))
        assert effects['oracle-stock']['regressions'] == 0
        result[role] = dict(absolute=absolute,contrasts=effects)
    mix = {a+'-'+b:.566*result['new']['contrasts'][a+'-'+b]['pp'] +
                    .434*result['returning']['contrasts'][a+'-'+b]['pp'] for a,b in PAIRS}
    eligible = (mix['causal-stock'] >= THRESHOLD and all(
        v['contrasts']['causal-stock']['paired80'][0] > 0 for v in result.values())
        and result['new']['contrasts']['causal-fixed']['paired80'][0] > 0)
    Path('result.json').write_text(json.dumps(dict(result=result,mix_pp=mix,
        illustrative_required_pp=THRESHOLD,eligible=eligible,receipt=ref.sha(Path('before_labels.json'))),
        indent=2)+'\n')
    print(mix,'eligible',eligible,flush=True)


if __name__ == '__main__':
    collect()
