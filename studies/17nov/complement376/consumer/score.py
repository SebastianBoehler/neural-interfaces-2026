"""Reuse validated parent controls and verify §376c per-order full-stage zero."""
import importlib.util
import json
import sys

import numpy as np
from context import ARMS, CHAIN, HERE, ORDERS, PARENT, WORK, require, pin, sha

sys.path.insert(0,str(CHAIN/'studies/14nov/newmember'))
import score_s_v66p as s


def collect():
    meta = json.loads((HERE/'parent.json').read_text())
    pin(HERE/'parent.npz',meta['sha'])
    pin(PARENT/'result.json',meta['result'])
    pin(PARENT/'runtime.sha256',meta['runtime'])
    with np.load(HERE/'parent.npz') as z:
        pred = {tuple(k.split('|')):z[k] for k in z.files}
    with np.load(CHAIN/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        people = z['person|all_loso']
        ns = len(set(z['sess|all_loso']))
    remote = s.remote_manifest(CHAIN/'runs/representation_inputs/streams.sha256')
    cache = {}
    from s281_routing import verify_archive_file
    s.verify_package(CHAIN/'runs/sealed/final_zips/v66_cp1_cap250.zip',
                     '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed')
    for order in ORDERS:
        archive = CHAIN/f'runs/s281_v66_reference/s242c250c_{order}.npz'
        h = verify_archive_file(archive,archive.parent/'sha256.remote.txt')
        s.verify_route(json.loads((CHAIN/f'runs/s281_v66_route/{order}.json').read_text()),order,h,s.BASES,ns)
        arms = ('zero_repr',*ARMS) if order!='within0' else ('control','zero_repr',*ARMS)
        for arm in arms:
            proof = CHAIN/f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json'
            s.verify_streams(json.loads(proof.read_text()),arm,order,remote)
            for base in s.BASES:
                output = s.load_output(CHAIN,arm,order,base,len(people),cache)
                if arm in ('control','zero_repr'):
                    s.verify_zero(pred['control',order,base],output)
                else:
                    pred[arm,order,base] = output
    return pred,people


def main():
    pred,people = collect()
    metric = WORK/'ni26_raw312_cpu_20261117_v2/experiments/protocol.py'
    pin(metric,'390fd6bea0ddc9c53e34fdabb094b1a269ff45b07b4a5542a8135b44d5314a2c')
    spec = importlib.util.spec_from_file_location('metric376c',metric)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    with np.load(CHAIN/'data/proteus_windows.npz') as z:
        meta = {k:z[k] for k in ('subject','session','interface')}
        y = z['y']
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    require(np.array_equal(people,[s.rsplit('/',1)[-1] for s in meta['subject']]),'row ownership mismatch')
    result = {}
    for label,bases in [('B2_pool',['B2_pool']),('six_bases',s.BASES),('individual',[])]:
        arms = ('RF','O','E','C','S','A','G') if not bases else ('control','RF','O','E','C','S','A','G')
        values = {a:np.empty((8,10)) for a in arms}
        counts = np.empty(10,int)
        for a in arms:
            for j,o in enumerate(ORDERS):
                if bases:
                    ys = [pred[a,o,b].argmax(1) for b in bases]
                else:
                    root = CHAIN if a in ARMS else PARENT/'chain'
                    fam = ('n376c' if a in ARMS else 'n376') + a.lower()
                    with np.load(root/f'runs/representation_streams/ttt18{o}_{fam}_base-all_loso-ensemble2.npz') as z:
                        ys = [z['proba'].argmax(1)]
                for i,p in enumerate(s.PEOPLE):
                    mask = people == p
                    rows = [protocol.score(y[mask],yp[mask],meta,mask) for yp in ys]
                    values[a][j,i] = 100*np.mean([r['balanced_accuracy_cell_mean'] for r in rows])
                    counts[i] = rows[0]['n_cells']
        contrasts = [('A','O'),('G','O'),('G','A'),('A','E'),('G','E')]
        if bases:
            contrasts += [('A','control'),('G','control')]
        for a,b in contrasts:
            values[a+'-'+b] = values[a]-values[b]
        result[label] = {k:dict(official=s.cell_weighted_summary(v,ORDERS,counts),
                                 people_order=s.summarize(v,ORDERS,s.PEOPLE)) for k,v in values.items()}
    old = json.loads((PARENT/'result.json').read_text())
    for label,group in result.items():
        for arm in ('O','RF','E','C','S'):
            require(abs(group[arm]['official']['mean']-old[label][arm]['official']['mean'])<1e-9,'parent score drift')
    with (HERE/'result.json').open('x') as f:
        json.dump(result,f,indent=2)


if __name__ == '__main__':
    main()
