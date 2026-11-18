"""Score paired external label selectivity without fitting a target readout."""
from pathlib import Path
import json
import numpy as np

BASE = Path('/mnt/lustre/work/gehler/gfh527')


def selectivity(pred, labels, runs):
    response = (pred == 2).astype(int) - (pred == 1).astype(int)
    cells = []
    for run in sorted(set(runs)):
        yes, no = (runs == run) & (labels == 1), (runs == run) & (labels == 0)
        assert yes.sum() == no.sum() == 10
        cells.append(100 * (response[..., yes].mean(-1) - response[..., no].mean(-1)))
    return np.mean(cells, axis=0)


def summary(values, draws):
    return [float(values.mean()), *np.quantile(values[draws].mean(1), [.1,.9]).tolist()]


def main():
    package = BASE/'ni26_select415_20261118'
    ref = json.loads((package/'result.json').read_text())['W']
    with np.load(BASE/'neural-interfaces-2026/data/shin2018/windows.npz') as data:
        labels, runs = data['y'], data['run']
    rows = {}
    for p in range(1,27):
        measures = {}
        with np.load(f'person{p}.npz') as z:
            for seed in (415,1415):
                idx = z[str(seed)]
                for k in z.files:
                    if not k.startswith(str(seed)+'|'):
                        continue
                    arm = k.split('|')[1]
                    measures.setdefault(arm, []).append(selectivity(z[k], labels[idx], runs[idx]))
                for arm in ('old','cap'):
                    bank = json.loads((package/f'{arm}_{6 + (p-1)//5}.json').read_text())
                    [r] = [r for r in bank if r['seed']==seed and r['person']==p]
                    assert np.array_equal(r['raw'],idx)
                    measures.setdefault(arm, []).append(selectivity(np.array(r['labels']), labels[idx], runs[idx]))
        for k,v in measures.items():
            assert len(v)==2
            rows.setdefault(k, []).append(np.mean(v,axis=0))
    rows = {k:np.asarray(v) for k,v in rows.items()}
    np.testing.assert_allclose(rows['cap']-rows['old'], ref['per_person'], atol=1e-10, rtol=0)
    draw = np.random.default_rng(418).integers(26,size=(10000,26))
    contrasts = {}
    for seed in (382,1382):
        j,k,b = [rows[f'{a}{seed}'] for a in ('J','K','blank')]
        for name,delta in [('real_minus_blank',j[:,0]-b[:,0]),('real_minus_native',j[:,0]-rows['native'][:,0]),
                           ('real_minus_joint_explicit',j[:,0]-k[:,0]),('real_horizon',j[:,0]-j[:,1]),
                           ('explicit_horizon',k[:,0]-k[:,1])]:
            contrasts[f'{seed}_{name}'] = summary(delta,draw)
    contrasts['native_minus_population']=summary(rows['native'][:,0]-rows['pop'][:,0],draw)
    data=dict(arms={k:a.mean(0).tolist() for k,a in rows.items()}, contrasts=contrasts,
              n_people=26,orders=[415,1415])
    Path('result.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    main()
