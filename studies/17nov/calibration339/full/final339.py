"""Original complete current-system scorer, then separate individual-member metrics."""
import json
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
ROOT=Path(os.environ['NI26_WS']);sys.path.insert(0,str(ROOT/'studies/14nov/newmember'))
import score_s_v66p as s
ARMS=('RF','RB','RV','CF','CB','CV')


def main():
    # Original complete provenance/zero/order/package guards BEFORE any outer metric.
    with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:n=len(z['sess|all_loso']);sessions=len(np.unique(z['sess|all_loso']))
    outputs=s.collect(ROOT,n,sessions)
    old=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_chain_20261117')
    if s.sha256(old/'results.txt')!='9e82b8082948702b600687b455e9efcd027b1698ae21955deb9e5faa0196ae34':raise ValueError('original323completechainchanged')
    code="import sys,numpy as np;from pathlib import Path;root=Path("+repr(str(old))+ ");sys.path.insert(0,str(root/'studies/14nov/newmember'));import score_s_v66p as s;z=np.load(root/'runs/replay_data/bases.npz',allow_pickle=True);s.collect(root,len(z['sess|all_loso']),len(np.unique(z['sess|all_loso'])));print('original323completeguardsOK')"
    subprocess.run([sys.executable,'-B','-I','-c',code],check=True)
    for a,b in (('control','control'),('RF','P0F'),('RB','P0C')):
        for o in s.ORDERS:
            for base in s.BASES:
                with np.load(old/f'runs/s281_v66_cp/{b}_{o}_{base}.npz') as z:
                    if not np.array_equal(outputs[a,o,base].argmax(1),z['scores'].argmax(1)):raise ValueError('raw323fullchainendpointchanged')
    print('FULLSYSTEMDEV — original complete current-system scorer; sixdependentbases',flush=True)
    s.main()
    with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:y=z['y|all_loso'];person=z['person|all_loso'];cell=z['cell|all_loso']
    counts=np.array([len(np.unique(cell[person==p])) for p in s.PEOPLE]);result={}
    for a in ARMS:
        values=[]
        for o in s.ORDERS:
            p=ROOT/f'runs/representation_streams/ttt18{o}_n339{a.lower()}_base-all_loso-ensemble2.npz'
            with np.load(p) as z:proba=z['proba']
            values.append(s.person_accuracy(proba,y,cell,person,s.PEOPLE))
        arr=np.stack(values);result[a]=dict(people_order=s.summarize(arr,s.ORDERS,s.PEOPLE),official_equal_cell=s.cell_weighted_summary(arr,s.ORDERS,counts))
    (ROOT/'individual339.json').write_text(json.dumps(dict(scope='INDIVIDUALMODEL actualtwo-seedmean;NOTfullsystem/SEALED',metrics=result),indent=2))
    print('INDIVIDUALMODEL metrics separately in individual339.json; SEALED notmeasured',flush=True)


if __name__=='__main__':main()
