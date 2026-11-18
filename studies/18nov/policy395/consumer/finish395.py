"""Stage independently verified §395 ensembles and apply the validated §376 full scorer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from setup_chain import HERE,ROOT,WORK,ARMS,CONTRASTS,owned

ACTOR=WORK/'ni26_policy395_actor_v1_20261118'
HIST=WORK/'ni26_calibration339_setup_v1_20261117'
sys.path.insert(0,str(ROOT/'studies/14nov/newmember'))
import score_s_v66p as scorer


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def check(ok,text):
    if not ok:
        raise ValueError(text)


def stage():
    assert sha(ACTOR/'before_labels.json')=='ac488cf9addbb5747e77ff27a66cc8ac9aa866305c9a88fce8aaeecf8507257f'
    assert sha(HIST/'validation.json')=='d0386068e179d9491891897ce87f4e312c6db161014cd45451cfdfaee22c447a'
    before=json.loads((ACTOR/'before_labels.json').read_text())
    hist=json.loads((HIST/'validation.json').read_text())
    dest=ROOT/'runs/representation_streams'
    lock=ROOT/'runs/representation_inputs'
    dest.mkdir();lock.mkdir()
    with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        people=z['person|all_loso']
    n=len(people)
    assert n==2655 and set(people)==set(scorer.PEOPLE)
    inputs={str(p):sha(p) for p in (ACTOR/'before_labels.json',HIST/'validation.json')}
    values={(a,o):np.full((n,3),np.nan) for a in ARMS for o in scorer.ORDERS}
    names={'RB':'native','L':'learned','M':'mean','D':'donor'}
    for p in scorer.PEOPLE:
        path=ACTOR/(p+'.npz')
        assert sha(path)==before['outputs'][p]
        inputs[str(path)]=sha(path)
        with np.load(path) as z:
            rows=np.flatnonzero(people==p)
            assert np.array_equal(rows,z['index'])
            for (a,o),v in values.items():
                v[rows]=np.mean([z[f'{s}|{names[a]}|{o}'] for s in (0,1)],0)
    hashes={}
    for o in scorer.ORDERS:
        p=HIST/'streams'/f'ttt18{o}_n339rb_base-all_loso-ensemble2.npz'
        assert sha(p)==hist['streams'][p.name]
        with np.load(p) as z:
            assert np.array_equal(values['RB',o],z['proba'])
        assert len({hashlib.sha256(values[a,o].tobytes()).hexdigest() for a in ARMS})==len(ARMS)
        for a in ARMS:
            v=values[a,o]
            assert np.isfinite(v).all() and (v>=0).all() and np.allclose(v.sum(1),1)
            family='n395'+a.lower()
            p=dest/f'ttt18{o}_{family}_base-all_loso-ensemble2.npz'
            np.savez_compressed(p,index=np.arange(n),proba=v,metadata=json.dumps(
                dict(seeds=[0,1],operation='arithmetic_probability_mean',family=family,order=o)))
            hashes[p.name]=sha(p)
    manifest=lock/'streams.sha256'
    manifest.write_text(''.join(f'{v}  {k}\n' for k,v in sorted(hashes.items())))
    (lock/'lock.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest),producer_inputs_sha256=inputs)))


def score():
    text=owned('finish_v2.py','d2a8cc56db494c5d711c424fd0e6b73cb3ed7c10e85de89eb923ac4f5f32921d')
    code=text[text.index('def score():'):text.index("if __name__ == '__main__':")]
    assert code.count('n376')==1 and code.count("('control','RF')")==1
    code=code.replace('n376','n395').replace("('control','RF')","('control','RB')")
    env={**globals(),'importlib':importlib}
    exec(compile(code,'verified376_scorer395', 'exec'),env)
    env['score']()
    result=json.loads((HERE/'result.json').read_text())
    actor=json.loads((ACTOR/'result.json').read_text())
    assert sha(ACTOR/'result.json')=='fbf1caf6a6b96823aea5953a13e5b8267baa1d9b1cdc5470508be88e13ce039b'
    for a,n in [('RB','native'),('L','learned'),('M','mean'),('D','donor')]:
        assert abs(result['individual'][a]['official']['mean']-actor['scores']['ensemble|'+n])<1e-9


if __name__=='__main__':
    assert Path(scorer.__file__).resolve()==ROOT/'studies/14nov/newmember/score_s_v66p.py'
    {'stage':stage,'score':score}[sys.argv[1]]()
