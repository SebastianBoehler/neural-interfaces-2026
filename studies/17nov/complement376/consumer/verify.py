"""Lock nested producer inputs and validate complete independent original outputs."""
import json
import sys
import numpy as np
from context import ARMS, CHAIN, HERE, ORDERS, P0, PARENT, PROD, pin, require, sha


def stage():
    pin(PROD/'source.sha256','b335b253c9c579449f019b329386195079d88d8208c3d9929bdab098a9910339')
    pin(P0/'source.sha256','31482d1ac91efbdb54fdaff81fd5c8f62c44ad7f575ab8896d88e77e1beab843')
    with np.load(CHAIN/'runs/replay_data/bases.npz', allow_pickle=True) as z:
        people = z['person|all_loso']
    outputs = {(a,o):np.full((len(people),3),np.nan) for a in ARMS for o in ORDERS}
    inputs = {str(PROD/'source.sha256'):sha(PROD/'source.sha256'),str(P0/'source.sha256'):sha(P0/'source.sha256')}
    for person in sorted(set(people)):
        proof, oldproof = PROD/person/'proof.json', P0/person/'proof.json'
        m, old = [json.loads(p.read_text()) for p in (proof,oldproof)]
        archive, origin = PROD/person/'fold.npz',P0/person/'fold.npz'
        require(m['held']==person and m['parent_sha']==sha(oldproof) and
                m['parent_archive_sha']==old['archive_sha256']==sha(origin) and
                m['archive_sha256']==sha(archive) and m['labels_opened'] is False and
                m['source_sha256']==sha(PROD/'source.sha256') and m['teacher']==old['teacher'] and
                m['nested_bank_sha']=='d110fc231ca91b84e90cdaa680e3710cece7e7439f47e1d517f1fd8dcd978ba8',
                'fold producer/parent mismatch')
        for file in (proof,oldproof,archive,origin):
            inputs[str(file)] = sha(file)
        held = np.flatnonzero(people==person)
        with np.load(archive) as z, np.load(origin) as orig:
            require(np.array_equal(z['index'],held) and np.array_equal(orig['index'],held),'outer row order')
            for order in ORDERS:
                for arm in ('RF','O','E','C','S'):
                    key = arm+'|'+order
                    require(np.array_equal(z[key],orig[key]),'parent control changed')
                for arm in ARMS:
                    outputs[arm,order][held] = z[arm+'|'+order]
    dest, lock = (CHAIN/'runs'/n for n in ('representation_streams','representation_inputs'))
    dest.mkdir()
    lock.mkdir()
    manifests = {}
    for (arm,order),q in outputs.items():
        require(np.isfinite(q).all() and np.all(q>=0) and np.allclose(q.sum(1),1)
                and not np.array_equal(outputs['A',order],outputs['G',order]),'invalid/inactive probabilities')
        family = 'n376c'+arm.lower()
        name = f'ttt18{order}_{family}_base-all_loso-ensemble2.npz'
        meta = dict(family=family,order=order,seeds=[0,1],operation='arithmetic_probability_mean')
        np.savez_compressed(dest/name,index=np.arange(len(q)),proba=q,metadata=json.dumps(meta))
        manifests[name] = sha(dest/name)
    manifest = lock/'streams.sha256'
    manifest.write_text(''.join(f'{v}  {k}\n' for k,v in sorted(manifests.items())))
    (lock/'lock.json').write_text(json.dumps(dict(manifest_sha256=sha(manifest),producer_inputs_sha256=inputs)))


def parent():
    root = PARENT/'chain'
    pin(PARENT/'runtime.sha256','a0e38b986672e3820c23b90f1620df6a5f792893063dbfcc4dc26c6fcb8ae7f9')
    pin(PARENT/'finish_v2.py','d2a8cc56db494c5d711c424fd0e6b73cb3ed7c10e85de89eb923ac4f5f32921d')
    report = json.loads((PARENT/'result.json').read_text())
    require(np.isclose(report['B2_pool']['control']['official']['mean'],76.27933502146),'unqualified parent result')
    sys.path.insert(0,str(root/'studies/14nov/newmember'))
    import score_s_v66p as s
    with np.load(root/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        n = len(z['person|all_loso'])
        sessions = len(set(z['sess|all_loso']))
    pred = s.collect(root,n,sessions)
    keys = ('control','RF','O','E','C','S')
    np.savez_compressed(HERE/'parent.npz',**{'|'.join((a,o,b)):pred[a,o,b]
                        for a in keys for o in s.ORDERS for b in s.BASES})
    (HERE/'parent.json').write_text(json.dumps(dict(sha=sha(HERE/'parent.npz'),
        result=sha(PARENT/'result.json'),runtime=sha(PARENT/'runtime.sha256'))))


if __name__=='__main__':
    {'stage':stage,'parent':parent}[sys.argv[1]]()
