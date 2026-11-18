"""Score after all eight full actors have sealed their outputs."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from full import ARMS, OLD, ORDERS, PEOPLE, sha, source_manifest

sys.path.insert(0, str(OLD))
import score379 as metrics


def main():
    assert Path('replay.log').read_text().splitlines()[-1]=='OK'
    manifest = source_manifest(Path('source.sha256'))
    data = metrics.BASE/'data/proteus_windows.npz'
    assert sha(data)==metrics.DATA_SHA
    with np.load(data) as z:
        meta = {k:z[k] for k in ('subject','session','interface')}
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    people = np.array([p.split('/')[-1] for p in meta['subject']])
    banks, ages = metrics.controls_and_age(meta)
    assert source_manifest(metrics.METRICS/'source.sha256')=='9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd'
    sys.path.insert(0,str(metrics.METRICS))
    from measure import official
    metric = official()
    scorer_file = metrics.BASE/'studies/14nov/newmember/score_s_v66p.py'
    assert sha(scorer_file)=='ae387cd5a7b116fc1259001d4fe449ba1065b0cb5943c117d4f65ec87a0b40d4'
    spec = importlib.util.spec_from_file_location('paired440',scorer_file)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    seals = []
    for a in ARMS:
        banks[a] = np.full(ages.shape,-1,dtype=np.int8)
    for j,o in enumerate(ORDERS):
        path = Path(f'order_{j}.npz')
        card = json.loads(path.with_suffix('.json').read_text())
        assert card['source']==manifest and card['order']==o and card['output']==sha(path)
        assert not seals or card['folds']==seals[0]['folds']
        with np.load(path) as z:
            assert set(z.files)==set(ARMS)
            for a in ARMS:
                assert z[a].shape==(len(people),) and np.isin(z[a],(0,1,2)).all()
                banks[a][j]=z[a]
        assert np.array_equal(banks['stock'][j],banks['frozen'][j])
        seals.append(card)
    Path('before440.json').write_text(json.dumps(seals,indent=2))
    with np.load(data) as z:
        y = z['y']
    metrics.CONTRASTS = [('phase_motor',a) for a in ('stock','real_motor','phase_all')] + [
        ('phase_all','real_all'),('real_motor','stock'),('phase_all','stock')]
    result = metrics.summarize(metric,scorer,meta,y,people,banks,ages>=0,'all')
    cells = np.char.add(np.char.add(meta['stream'],'|'),meta['interface'])
    result['class_recalls'] = {}
    for a in ARMS:
        by_order = np.array([[100*np.mean([(d[(cells==c)&(y==k)]==k).mean() for c in np.unique(cells)])
                              for k in range(3)] for d in banks[a]])
        result['class_recalls'][a] = np.mean([by_order[:4].mean(0),by_order[4:6].mean(0),by_order[6:].mean(0)],0).tolist()
    result['before_sha'] = sha(Path('before440.json'))
    Path('result440.json').write_text(json.dumps(result,indent=2))
    print(result['scores'])


if __name__ == '__main__':
    main()
