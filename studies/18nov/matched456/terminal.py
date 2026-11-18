"""Apply already fitted terminal corrections to both stock generations."""
import json
import numpy as np
from stocks456 import CONSENSUS, MEMORY, log_probability, module, ref


def consensus_inputs(person, rows, ages, streams):
    owner = module(CONSENSUS, 'run', '8f64b6fe9bc0c9e9c565cae67760417fb1376ad60321118977cd2ad46ff99d1b')
    banks = [owner.seed_deltas(seed, person, rows)[0] for seed in owner.PINS]
    result = {}
    for name in ('memory', 'donor'):
        a, b = [d[name] for d in banks]
        prefix = '' if name == 'memory' else 'donor_'
        pa, pb = [np.stack([owner.prior_mean(d[i], ages[o][rows], streams) for i,o in enumerate(ref.ORDERS)]) for d in (a,b)]
        result.update({prefix+'consensus':owner.consensus(a,b), prefix+'mean':(a+b)/2,
                       prefix+'prior':owner.consensus(pa,pb),
                       prefix+'residual':owner.consensus(np.clip(a-pa,-1,1),np.clip(b-pb,-1,1))})
    return {arm:.1*v for arm,v in result.items()}


def historical_references(family, folder=MEMORY):
    if family == 'consensus':
        folder = CONSENSUS
        inputs = [(phase+'_decisions.npz', phase+'_before_labels.json','predictions') for phase in ('agreement','residual')]
    else:
        inputs = [(family+'_decisions.npz',family+'_decisions.json','decisions')]
    reference = {}
    for npz, receipt, key in inputs:
        path = folder / npz
        assert ref.sha(path) == json.loads((folder/receipt).read_text())[key]
        with np.load(path) as z:
            for k in z:
                if k in reference:
                    assert np.array_equal(reference[k], z[k])
                reference[k] = z[k]
    return reference


def build(family, meta, ages, historical, stock, smoke):
    if family.startswith('gain_'):
        seed = family.split('_')[1]
        combined = {}
        for actor in ('kernel','coherent','scattered'):
            arms = build(f'{actor}_{seed}', meta, ages, historical, stock, smoke)
            for (p,o,a),pred in arms.items():
                key = (p,o,a if a == 'stock' else actor+'_'+a)
                if key in combined:
                    assert np.array_equal(pred,combined[key])
                combined[key] = pred
        return combined
    folder = ref.WORK/'ni26_gain454_20261118' if family.split('_')[0] in ('coherent','scattered') else MEMORY
    refs = historical_references(family, folder)
    bank = {}
    for person in ref.PEOPLE[:1] if smoke else ref.PEOPLE:
        rows = np.flatnonzero(meta['person'] == person)
        if family == 'consensus':
            delta = consensus_inputs(person,rows,ages,meta['stream'][rows])
            delta['stock'] = np.zeros_like(delta['consensus'])
        else:
            kind, seed = family.split('_')
            owner = module(MEMORY, 'corrections', '528f76f9d69d56e4be714bb0cf6d48bbff665b46dad1f7eeddcd0cb1307e1e07')
            delta, _ = owner.load(folder, kind, int(seed), person, rows)
        for i,order in enumerate(ref.ORDERS):
            old = log_probability(historical[order][rows], ages[order][rows])
            new = log_probability(stock[order][rows], ages[order][rows])
            for arm, diff in delta.items():
                key = '|'.join((person, order, arm))
                assert np.array_equal((old+diff[i]).argmax(1),refs[key]),key
                pred = (new+diff[i]).argmax(1)
                if arm in ('source','stock'):
                    assert not np.any(diff[i]) and np.array_equal(pred,stock[order][rows].argmax(1))
                outarm = 'stock' if arm == 'source' else arm
                bank[person,order,outarm] = pred.astype(np.uint8)
    return bank
