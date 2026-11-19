"""Seal complete F26 held banks, then score equal cells with nested resampling."""
import json

import numpy as np

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, sha
from held import ACTOR, F16, paired
from reference_map import ARMS, HERE, KINDS, MODES
from replay_f13 import DATA, ORDER_SHA, ORDERS, ROOT, origin_report
from source_targets import TOL


F16_RESULT = 'e91c377bd9560d7c02f01adc9ccaa5e82b87d347927c19e17fd1d217780a803e'
CONTRASTS = (('car','hard'), ('car','sham'), ('sham','hard'))


def seal(subject,person,visits,stream):
    all_values = np.full((2,len(ARMS),len(ORDERS),len(subject)),-1,dtype=np.int8)
    manifest, summaries = [], []
    scorer = sha(HERE/'score.sha256')
    for seed in (0,1):
        for held in PEOPLE:
            path = HERE/'folds'/f'{held}_s{seed}.npz'
            digest = sha(path)
            if digest != path.with_suffix('.sha256').read_text().strip():
                raise ValueError('held F26 proof digest differs')
            rows = np.flatnonzero(person==held)
            required = dict(held=held,seed=seed,actor=ACTOR,scorer=scorer,data=DATA_SHA,origin=ORIGIN_SHA,
                            orders=list(ORDERS),order_sha=ORDER_SHA,held_labels_opened=False)
            with np.load(path,allow_pickle=False) as own,np.load(F16/'folds'/path.name,allow_pickle=False) as base:
                metadata = json.loads(str(own['metadata']))
                if (any(metadata[k]!=v for k,v in required.items()) or not np.array_equal(own['index'],rows) or
                        not np.array_equal(base['index'],rows) or set(metadata['evidence'])!=set(KINDS) or
                        any(not np.array_equal(own[f'hard_{m}'],base[f'hard_{m}']) for m in MODES)):
                    raise ValueError('held F26/F16 order or coverage differs')
                values = np.stack([own[a] for a in ARMS])
            if values.shape!=(len(ARMS),len(ORDERS),len(rows)) or not np.isin(values,(0,1,2)).all():
                raise ValueError('invalid F26 prediction bank')
            first = len(set(stream[rows]))*len(ORDERS)
            for kind,(cp,source) in paired(held,seed,False,subject,visits).items():
                p = metadata['evidence'][kind]
                expected = dict(receipt=sha(cp.with_suffix('.json')),checkpoint=source['checkpoint'],state=source['state'],
                                first_windows_exact=first,resets=2*first)
                if (any(p[k]!=v for k,v in expected.items()) or not 0 <= p['prefix_max_abs'] <= TOL or
                        (p['reference_calls'] != 0 if kind=='hard' else p['reference_calls'] <= p['resets'])):
                    raise ValueError('incorrect causal, active-map or fit-exclusion receipt')
                summaries.append(p)
            all_values[seed][:,:,rows] = values
            manifest.append(f'{digest}  folds/{path.name}\n')
    if (all_values<0).any() or not any(r['clips'] for r in summaries):
        raise ValueError('incomplete matched F26 matrix')
    lock = HERE/'before_labels.sha256'
    with lock.open('x') as out:
        out.writelines(sorted(manifest))
    return all_values,sha(lock),dict(first=sum(r['first_windows_exact'] for r in summaries),
                                     resets=sum(r['resets'] for r in summaries),
                                     prefix_max_abs=max(r['prefix_max_abs'] for r in summaries))


def score(values,labels,cells,people,dataset):
    import protocol
    import score_s_v66p as metrics

    if (protocol.__file__ != str(ROOT/'experiments/protocol.py') or
            metrics.__file__ != str(ROOT/'studies/14nov/newmember/score_s_v66p.py')):
        raise ValueError('wrong original whole-decoder metric')
    counts = np.array([len(set(cells[people==p])) for p in PEOPLE])
    schemes = [np.flatnonzero([o.startswith(s) for o in ORDERS]) for s in ('within','full','runs')]
    hard = F16/'whole_results.json'
    if sha(hard)!=F16_RESULT:
        raise ValueError('F16 score reference differs')
    reference = json.loads(hard.read_text())['seeds']
    results = {}
    for seed,vals in enumerate(values):
        absolute,individual,paired = {},{},{}
        for arm,p in zip(ARMS,vals,strict=True):
            person = np.stack([metrics.person_accuracy(np.eye(3)[z],labels,cells,people,PEOPLE) for z in p])
            official = np.asarray([100*protocol.score(labels,z,dataset,np.ones(len(z),bool))['balanced_accuracy_cell_mean'] for z in p])
            np.testing.assert_allclose(np.average(person,weights=counts,axis=1),official,atol=1e-10,rtol=0)
            absolute[arm] = float(np.mean([official[g].mean() for g in schemes]))
            individual[arm] = person
        for mode in MODES:
            key = f'hard_{mode}'
            if abs(absolute[key]-reference[str(seed)]['absolute'][key])>1e-10:
                raise ValueError('F16 hard score differs')
            for a,b in CONTRASTS:
                delta = individual[f'{a}_{mode}']-individual[f'{b}_{mode}']
                paired[f'{mode}_{a}_minus_{b}'] = dict(metrics.cell_weighted_summary(delta,ORDERS,counts),
                    people=dict(zip(PEOPLE,np.mean([delta[g].mean(0) for g in schemes],0).tolist())),
                    orders=dict(zip(ORDERS,np.average(delta,weights=counts,axis=1).tolist())))
        results[str(seed)] = dict(absolute=absolute,paired=paired)
    outcomes = {}
    for a,b in CONTRASTS:
        key = f'bounded_{a}_minus_{b}'
        first,second = (results[str(s)]['paired'][key]['interval'] for s in (0,1))
        outcomes[key] = ('supported' if min(first[0],second[0])>0 else
                         'refuted' if max(first[1],second[1])<=0 else 'undecided')
    return results,outcomes


def run():
    import protocol

    origin_report()
    with np.load(DATA,allow_pickle=False) as data:
        subject,visits = data['subject'],data['session']
    people = np.asarray([p.rsplit('/',1)[-1] for p in subject])
    stream = np.char.add(np.char.add(subject,'|'),visits)
    values,before,guards = seal(subject,people,visits,stream)
    dataset = protocol.load()
    np.testing.assert_array_equal(dataset['subject'],subject)
    np.testing.assert_array_equal(dataset['stream'],stream)
    cells = np.char.add(np.char.add(stream,'|'),dataset['interface'])
    seeds,outcome = score(values,dataset['y'],cells,people,dataset)
    path = HERE/'whole_results.json'
    with path.open('x') as out:
        json.dump(dict(scope='held-new WHOLE DEV',before_labels=before,guards=guards,actor=ACTOR,
                       scorer=sha(HERE/'score.sha256'),seeds=seeds,outcome=outcome),out,indent=2,sort_keys=True,allow_nan=False)
        out.write('\n')
    print(f'PASS F26 whole score {sha(path)}',flush=True)


if __name__=='__main__':
    run()
