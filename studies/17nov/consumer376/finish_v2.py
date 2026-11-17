"""Stage only verified outer-fold members, then score complete chain endpoints."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from setup_chain import ARMS, CONTRASTS, HERE, ROOT, WORK

PRODUCER = WORK/'ni26_complement376_v2_20261117'
PINS = {PRODUCER/'source.sha256':'31482d1ac91efbdb54fdaff81fd5c8f62c44ad7f575ab8896d88e77e1beab843',
        HERE/'runtime.sha256':'a0e38b986672e3820c23b90f1620df6a5f792893063dbfcc4dc26c6fcb8ae7f9'}
OLD = WORK/'ni26_calibration339_setup_v1_20261117'
VALID = OLD/'validation.json'
PINS[VALID] = 'd0386068e179d9491891897ce87f4e312c6db161014cd45451cfdfaee22c447a'
sys.path.insert(0,str(ROOT/'studies/14nov/newmember'))
import score_s_v66p as scorer


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def check(ok, detail):
    if not ok:
        raise ValueError(detail)


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def stage():
    old = json.loads(VALID.read_text())['streams']
    dest, locks = (ROOT/'runs'/d for d in ('representation_streams','representation_inputs'))
    dest.mkdir()
    locks.mkdir()
    with np.load(ROOT/'runs/replay_data/bases.npz', allow_pickle=True) as z:
        people = z['person|all_loso']
    rows = np.arange(len(people))
    check(len(rows) == 2655 and len(set(people)) == 10,'public scope')
    matrices = {(a,o):np.full((len(rows),3),np.nan) for a in ARMS for o in scorer.ORDERS}
    inputs = {str(p):h for p,h in PINS.items()}
    for p in sorted(set(people)):
        src, held = rows[people != p], rows[people == p]
        archive, proof = (PRODUCER/p/n for n in ('fold.npz','proof.json'))
        m = json.loads(proof.read_text())
        check(m['held'] == p and m['seeds'] == [0,1] and m['labels_opened'] is False and
              set(m['train_people']) == set(people)-{p} and m['train_rows'] == array_sha(src) and
              m['test_rows'] == array_sha(held) and m['source_sha256'] == PINS[PRODUCER/'source.sha256']
              and m['archive_sha256'] == sha(archive),'outer exclusion '+p)
        inputs.update({str(archive):m['archive_sha256'],str(proof):sha(proof)})
        inner = m['teacher']['inner_folds']
        check(set(inner) == set(people)-{p}, 'inner inventory')
        for q, info in inner.items():
            fit = src[people[src] != q]
            check(set(info['fit_people']) == set(people)-{p,q} and info['fit_rows'] == array_sha(fit)
                  and info['query_rows'] == array_sha(src[people[src] == q]), 'nested source '+p+q)
        with np.load(archive,allow_pickle=False) as z:
            check(np.array_equal(z['index'],held),'order alignment')
            for (a,o),v in matrices.items():
                v[held] = z[a+'|'+o]
    hashes = {}
    for o in scorer.ORDERS:
        oldname = f'ttt18{o}_n339rf_base-all_loso-ensemble2.npz'
        ref = OLD/'streams'/oldname
        check(sha(ref) == old[oldname], 'RF historical hash')
        with np.load(ref) as z:
            check(np.array_equal(z['proba'],matrices['RF',o]),'raw §339 endpoint')
        check(len({array_sha(matrices[a,o]) for a in ARMS}) == len(ARMS),'inactive comparison')
        for a in ARMS:
            v = matrices[a,o]
            check(np.isfinite(v).all() and (v >= 0).all() and np.allclose(v.sum(1),1,atol=1e-6), 'posterior')
            family = 'n376'+a.lower()
            path = dest/f'ttt18{o}_{family}_base-all_loso-ensemble2.npz'
            np.savez_compressed(path,index=rows,proba=v,metadata=json.dumps(dict(seeds=[0,1],
                operation='arithmetic_probability_mean',family=family,order=o)))
            hashes[path.name] = sha(path)
    manifest = locks/'streams.sha256'
    manifest.write_text(''.join(f'{h}  {n}\n' for n,h in sorted(hashes.items())))
    (locks/'lock.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest),producer_inputs_sha256=inputs)))


def score():
    with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        people, sess = z['person|all_loso'], z['sess|all_loso']
    predictions = scorer.collect(ROOT,len(people),len(set(sess)))
    metricfile = WORK/'ni26_raw312_cpu_20261117_v2/experiments/protocol.py'
    check(sha(metricfile) == '390fd6bea0ddc9c53e34fdabb094b1a269ff45b07b4a5542a8135b44d5314a2c','metric')
    spec = importlib.util.spec_from_file_location('protocol376',metricfile)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    for a in ('control','RF'):
        for o in scorer.ORDERS:
            for b in scorer.BASES:
                with np.load(WORK/f'ni26_calibration339_chain_v1_20261117/runs/s281_v66_cp/{a}_{o}_{b}.npz') as z:
                    check(np.array_equal(predictions[a,o,b].argmax(1),z['scores'].argmax(1)),'historical endpoint')
    with np.load(ROOT/'data/proteus_windows.npz') as z:
        meta = {k:z[k] for k in ('subject','session','interface')}
        y = z['y']
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    check(np.array_equal(people,[s.split('/')[-1] for s in meta['subject']]),'metric row owners')
    result = {}
    for label,bases in [('B2_pool',['B2_pool']),('six_bases',scorer.BASES),('individual',[])]:
        arms = ARMS if not bases else ('control',*ARMS)
        scores = {a:np.zeros((8,10)) for a in arms}
        counts = np.zeros(10,dtype=int)
        for a in arms:
            for oi,o in enumerate(scorer.ORDERS):
                if bases:
                    ys = [predictions[a,o,b].argmax(1) for b in bases]
                else:
                    with np.load(ROOT/f'runs/representation_streams/ttt18{o}_n376{a.lower()}_base-all_loso-ensemble2.npz') as z:
                        ys = [z['proba'].argmax(1)]
                for pi,p in enumerate(scorer.PEOPLE):
                    mask = people == p
                    for pred in ys:
                        v = protocol.score(y[mask],pred[mask],meta,mask)
                        scores[a][oi,pi] += 100*v['balanced_accuracy_cell_mean']/len(ys)
                        counts[pi] = v['n_cells']
        for a,b in CONTRASTS + ([(a,'control') for a in ARMS] if bases else []):
            scores[a+'-'+b] = scores[a]-scores[b]
        result[label] = {k:dict(official=scorer.cell_weighted_summary(v,scorer.ORDERS,counts),
                                people_order=scorer.summarize(v,scorer.ORDERS,scorer.PEOPLE)) for k,v in scores.items()}
    check(abs(result['B2_pool']['control']['official']['mean']-76.27933502146)<1e-7,'baseline')
    (HERE/'result.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    for path,expected in PINS.items():
        check(sha(path) == expected,f'bad pin {path}')
    check(Path(scorer.__file__).resolve() == ROOT/'studies/14nov/newmember/score_s_v66p.py','wrong scorer')
    {'stage':stage,'score':score}[sys.argv[1]]()
