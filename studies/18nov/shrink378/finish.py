"""Verify source-only estimators, archive identity and the complete historical consumer."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from setup_chain import WORK, HERE, ROOT, ARMS, CONTRASTS, owned

PRODUCER = WORK/'ni26_shrink378_v3_20261118'
PRODUCER_SHA = '418012cd84e6fd6cd402e75e2577ecd0f6e82a106f3c478d1a9ff8365ae5daef'
OLD = WORK/'ni26_calibration339_setup_v1_20261117'
VALID = OLD/'validation.json'
VALID_SHA = 'd0386068e179d9491891897ce87f4e312c6db161014cd45451cfdfaee22c447a'
sys.path.insert(0,str(ROOT/'studies/14nov/newmember'))
import score_s_v66p as scorer


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def stage():
    old = json.loads(VALID.read_text())['streams']
    dest, locks = (ROOT/'runs'/d for d in ('representation_streams','representation_inputs'))
    dest.mkdir()
    locks.mkdir()
    with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        people = z['person|all_loso']
    rows = np.arange(len(people))
    check(len(rows)==2655 and set(people)==set(scorer.PEOPLE),'scope')
    matrices = {(a,o):np.full((len(rows),3),np.nan) for a in ARMS for o in scorer.ORDERS}
    inputs = {str(PRODUCER/'source.sha256'):PRODUCER_SHA, str(VALID):VALID_SHA}
    for p in scorer.PEOPLE:
        record = PRODUCER/p/'proof.json'
        archive = PRODUCER/p/'fold.npz'
        src,held = rows[people!=p],rows[people==p]
        m = json.loads(record.read_text())
        check(m['held']==p and m['labels_opened'] is False and m['seeds']==[0,1] and
              m['source_sha256']==PRODUCER_SHA and m['archive_sha256']==sha(archive) and
              m['test_rows']==digest(held) and m['train_rows']==digest(src) and
              set(m['train_people'])==set(people)-{p},'source exclusion '+p)
        inputs.update({str(record):sha(record),str(archive):sha(archive)})
        with np.load(archive) as z:
            check(np.array_equal(z['index'],held),'order alignment')
            for (a,o),v in matrices.items():
                v[held] = z[a+'|'+o]
    hashes = {}
    for o in scorer.ORDERS:
        for a in ('RF','RV'):
            name = f'ttt18{o}_n339{a.lower()}_base-all_loso-ensemble2.npz'
            path = OLD/'streams'/name
            check(sha(path)==old[name],'historical archive pin')
            with np.load(path) as z:
                check(np.array_equal(z['proba'],matrices[a,o]),'inactive historical path '+a)
        check(len({digest(matrices[a,o]) for a in ARMS})==len(ARMS),'inactive intervention')
        for a in ARMS:
            value=matrices[a,o]
            check(np.isfinite(value).all() and np.allclose(value.sum(1),1) and (value>=0).all(),'posterior')
            family='n378'+a.lower()
            path=dest/f'ttt18{o}_{family}_base-all_loso-ensemble2.npz'
            np.savez_compressed(path,index=rows,proba=value,metadata=json.dumps(dict(seeds=[0,1],
                operation='arithmetic_probability_mean',family=family,order=o)))
            hashes[path.name]=sha(path)
    manifest=locks/'streams.sha256'
    manifest.write_text(''.join(f'{h}  {n}\n' for n,h in sorted(hashes.items())))
    (locks/'lock.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest),producer_inputs_sha256=inputs)))


def score():
    # The same collector verifies endpoint identity and computes official matched intervals.
    text=owned('finish_v2.py','d2a8cc56db494c5d711c424fd0e6b73cb3ed7c10e85de89eb923ac4f5f32921d')
    block=text[text.index('def score():'):text.index("if __name__ == '__main__':")]
    assert block.count('n376')==1 and block.count("('control','RF')")==1
    block=block.replace('n376','n378').replace("('control','RF')","('control','RF','RV')")
    import importlib.util
    env={**globals(),'importlib':importlib}
    exec(compile(block,'verified376_collector_extended_to_RV','exec'),env)
    env['score']()


if __name__=='__main__':
    assert sha(PRODUCER/'source.sha256')==PRODUCER_SHA and sha(VALID)==VALID_SHA
    assert Path(scorer.__file__).resolve() == ROOT/'studies/14nov/newmember/score_s_v66p.py'
    {'stage':stage,'score':score}[sys.argv[1]]()
