"""Reproduce the §376 source weights, then measure whether the proxy locates actual errors."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

BASE = Path('/mnt/lustre/work/gehler/gfh527')
PARENT = BASE/'ni26_complement376_v2_20261117'
sys.path.insert(0,str(PARENT))
from produce import src
from nested import balanced, digest, features, require
from fidelity_score import score

PIN = '31482d1ac91efbdb54fdaff81fd5c8f62c44ad7f575ab8896d88e77e1beab843'
ARCH = BASE/'ni26_selector363_v1_20261117'


def predict(z, train, query, group, y):
    feat = z[train]
    mu, sd = feat.mean(0), feat.std(0).clip(1e-4)
    fit = (feat-mu)/sd
    w = balanced(group[train])
    require(np.all(y[train] >= 0), 'attempt to use unobserved label')
    target = np.eye(3)[y[train]]
    prior = np.average(target, axis=0, weights=w)
    gram = (fit.T*w)@fit
    gram.flat[::len(gram)+1] += .1*len(fit)
    beta = np.linalg.solve(gram,(fit.T*w)@(target-prior))
    return softmax(((z[query]-mu)/sd)@beta+prior,axis=1)


def weight_pins(q, group, y):
    w = {'O':balanced(group)}
    w.update({k:np.empty(len(group)) for k in 'ECS'})
    err = 1+2*(q.argmax(1)!=y)
    rng = np.random.default_rng(376)
    for g in np.unique(group):
        rows = np.flatnonzero(group==g)
        val = err[rows]/err[rows].mean()*w['O'][rows]
        w['E'][rows] = val
        w['C'][rows[np.argsort(1-q[rows].max(1),kind='stable')]] = np.sort(val)
        w['S'][rng.permutation(rows)] = val
    return {k:digest(v) for k,v in w.items()}


def main():
    require(src.sha(PARENT/'source.sha256') == PIN,'§376 source changed')
    for line in (PARENT/'source.sha256').read_text().splitlines():
        h,n = line.split(maxsplit=1)
        require(src.sha(PARENT/n)==h,'§376 code changed')
    src.check()
    with np.load(src.data323.DATA,allow_pickle=False) as z:
        meta = {k:z[k] for k in ('subject','session','interface')}
        feat = features(z['X'])
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    people = np.array([s.split('/')[-1] for s in meta['subject']])
    sorted_people = sorted(set(people))
    output, traces = {}, {}
    outer_prob = np.full((len(people),3),np.nan)
    for held in sorted_people:
        rows = np.flatnonzero(people!=held)
        query = np.flatnonzero(people==held)
        y = np.full(len(people),-1,int)
        y[rows] = src.source_labels.read_training_labels(src.data323.DATA,rows,len(people))
        group = np.array(['|'.join((s,i,str(c))) for s,i,c in zip(meta['stream'],meta['interface'],y)])
        inner = np.full((len(people),3),np.nan)
        path = PARENT/held/'proof.json'
        old = json.loads(path.read_text())
        require(old['source_sha256'] == PIN and old['train_rows'] == digest(rows) and old['test_rows'] == digest(query)
                and old['labels_opened'] is False, 'parent outer split')
        for cal in sorted(set(people[rows])):
            training = rows[people[rows]!=cal]
            apply = rows[people[rows]==cal]
            previous = old['teacher']['inner_folds'][cal]
            require(previous['fit_rows'] == digest(training) and previous['query_rows'] == digest(apply)
                    and previous['fit_features'] == digest(feat[training]), 'parent inner split/feature')
            inner[apply] = predict(feat,training,apply,group,y)
        require(weight_pins(inner[rows],group[rows],y[rows])==old['teacher']['weights'],
                'reconstructed original treatment weights differ')
        outer_prob[query] = predict(feat,rows,query,group,y)
        output[held+'_inner'] = inner[rows]
        traces[held] = dict(fit_rows=digest(rows),query_rows=digest(query),proof_sha=src.sha(path),
                            exact_inner_weight_shas=True,
                            probability_bytes_exact=digest(inner[rows])==old['teacher']['source_only_probabilities_sha'])
    require(np.isfinite(outer_prob).all(),'incomplete held predictions')
    np.savez_compressed('bank.npz',outer=outer_prob,**output)
    with open('before_labels.json','x') as f:
        json.dump(dict(traces=traces,bank_sha=src.sha(Path('bank.npz')),source_sha=src.sha(Path('source.sha256')),
                       outer_scoring_labels_opened=False),f,indent=2)
    p = ARCH/'predictions.npz'
    require(src.sha(p)=='df668bff888f4433eafa7d954246218c21762487518a20c78b47ab6b8aa6a0af','base archive drift')
    with np.load(p) as z:
        original = z['base']
    y = src.source_labels.read_training_labels(src.data323.DATA,np.arange(len(people)),len(people))
    result = score(outer_prob,original,meta,people,y)
    result['proof_sha'] = src.sha(Path('before_labels.json'))
    with open('result.json','x') as f:
        json.dump(result,f,indent=2)


if __name__ == '__main__':
    main()
