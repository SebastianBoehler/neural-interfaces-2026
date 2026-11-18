"""Commit predictions before evaluating the four held people and paired interfaces."""
import json
import sys
from pathlib import Path
import numpy as np
from model import decode, emissions, parts, powers, temperature

BASE = Path('/mnt/lustre/work/gehler/gfh527')
sys.path.insert(0, str(BASE/'ni26_fusion383_v1_20261118'))
from experiment import interval
from fusion_inputs import CHAIN, COHORT, member_bank, official, probability, restore, sha
sys.path.insert(0, str(BASE/'ni26_burst407_v1_20261118'))
from experiment407 import source
DATA = CHAIN/'data/proteus_windows.npz'


def repeatability(x, y, meta):
    people = np.unique(meta['person'])
    maps = []
    for iface in ('Graz', 'BrainHero'):
        group = []
        for p in people:
            mask = (meta['person'] == p) & (meta['interface'] == iface)
            visits = []
            for s in np.unique(meta['session'][mask]):
                rows = mask & (meta['session'] == s)
                means = [x[rows & (y == c)].mean(0) for c in range(3)]
                visits.append(np.subtract(means[1:], means[0]))
            group.append(np.mean(visits, axis=0))
        group = np.array(group)
        group -= group.mean(0)
        maps.append(group / np.linalg.norm(group, axis=2, keepdims=True))
    similarity = np.einsum('icd,jcd->ijc', *maps)
    own = similarity[np.arange(len(people)), np.arange(len(people))]
    return similarity, own - (similarity.sum(1) - own) / (len(people) - 1)


def study(smoke, calibrated):
    sc, scores, meta, ages = restore()
    with np.load(DATA) as z:
        names = z['ch_names'].tolist()
        raw = z['X']
        power = np.concatenate([powers(raw[i:i+32]) for i in range(0, len(raw), 32)])
        np.testing.assert_allclose(powers(-raw[:1]), power[:1])
        del raw
    rows = np.flatnonzero(~np.isin(meta['person'], COHORT))
    assert set(meta['person'][rows]) == {'P04','P05','P06','P07','P08','P10'}
    labels = source.read_training_labels(DATA, rows, len(power))
    weights = source.design_weights(meta['stream'][rows], meta['interface'][rows], labels)
    models, calibration = [], {}
    for seed in (None, 417):
        partition = parts(power, names, seed)
        factor, loss = temperature(partition, rows, labels, weights, meta['person'][rows]) if calibrated else (1, [])
        models.append(factor * emissions(partition, rows, labels, weights))
        calibration[str(seed)] = dict(factor=factor, source_loss=loss)
    preds, state = {}, {}
    for p in COHORT[:1] if smoke else COHORT:
        idx = np.flatnonzero(meta['person'] == p)
        assert len(set(meta['stream'][idx])) == 1
        bank = member_bank(382, p, idx, sc.ORDERS)
        for o in sc.ORDERS[:1] if smoke else sc.ORDERS:
            age = ages[o][idx]
            pos = np.argsort(age)
            np.testing.assert_array_equal(age[pos], np.arange(len(idx)))
            frozen = probability(scores['control', o, 'B2_pool'][idx], age)
            np.testing.assert_array_equal(frozen.argmax(1), bank['frozen',o].argmax(1))
            preds[p,o,'full','zero'] = frozen.argmax(1)
            for tag, model in zip(('', 'random_'), models):
                logp = model[idx[pos]]
                dec, prior = decode(logp)
                np.testing.assert_array_equal(decode(logp[:29])[0]['latent'], dec['latent'][:29])
                np.testing.assert_array_equal(dec['equal'][0], dec['latent'][0])
                assert np.ptp(prior) > 0 and np.isfinite(logp).all()
                assert np.max(abs(dec['equal'] - dec['latent'])) > 0
                state[p+'|'+o+'|'+tag] = [float(prior.min()),float(prior.max()),float(prior[-1])]
                for name, value in dec.items():
                    prob = value[np.argsort(pos)]
                    preds[p,o,'whole',tag+name] = prob.argmax(1)
                    preds[p,o,'full',tag+name] = (.9*frozen + .1*prob).argmax(1)
    if smoke:
        return
    np.savez_compressed('predictions.npz', **{'|'.join(k):v for k,v in preds.items()})
    Path('before_labels.json').write_text(json.dumps(dict(outputs=sha(Path('predictions.npz')),
        calibration=calibration, source=sha(Path('source.sha256')), fit=sorted(set(meta['person'][rows])), states=state)))
    with np.load(DATA) as z:
        y = z['y']
    metric = official()
    scores = {}
    for (p,o,kind,name), v in preds.items():
        mask = meta['person'] == p
        s = metric(y[mask], v, meta, mask)
        assert s['n_cells'] == 2
        scores.setdefault((kind,name),np.empty((4,8)))[COHORT.index(p),sc.ORDERS.index(o)] = 100*s['balanced_accuracy_cell_mean']
    assert abs(interval(scores['full','zero'])['pp'] - 81.21561492579204) < 1e-8
    for kind in ('whole','full'):
        for a in ('native','reverse','equal','random_latent'):
            scores[kind,'latent-'+a] = scores[kind,'latent'] - scores[kind,a]
    scores['full','latent-zero'] = scores['full','latent'] - scores['full','zero']
    result = {'scores':{'|'.join(k):interval(v) for k,v in scores.items()}}
    rng = np.random.default_rng(417)
    draws = rng.integers(10, size=(5000,10))
    sims, true = repeatability(parts(power,names)[1],y,meta)
    controls = [repeatability(parts(power,names,s)[1],y,meta)[1] for s in range(417,433)]
    diagnostics = {}
    for arm, arr in [('own_minus_mismatch',true), ('true_minus_random',true-np.mean(controls,0))]:
        for c, val in [('SUB',arr[:,0]),('WORD',arr[:,1]),('both',arr.mean(1))]:
            diagnostics[arm+'|'+c] = [float(val.mean()), *np.percentile(val[draws].mean(1),[10,90]).tolist()]
    own = sims[np.arange(10),np.arange(10)].mean()
    null = [sims[np.arange(10),rng.permutation(10)].mean() for _ in range(5000)]
    result.update(calibration=calibration,diagnostics=diagnostics,person_swap_p=(1+int(np.sum(null>=own)))/5001,
                  random_axes=[float(x.mean()) for x in controls],proof=sha(Path('before_labels.json')))
    Path('result.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    study(sys.argv[1] == 'smoke', sys.argv[2:] == ['calibrate'])
