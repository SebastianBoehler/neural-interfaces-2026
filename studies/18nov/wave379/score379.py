"""Score five complete independent decoders against two frozen complete system traces."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

WORK = Path('/mnt/lustre/work/gehler/gfh527')
BASE = WORK/'ni26_s318_coral_fbl_20261117'
METRICS = WORK/'ni26_soft_csp361_v1_20261117'
CONTROL = WORK/'ni26_joint373_v2_20261117/controls.npz'
ORDER_SHA = '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
CONTROL_SHA = '3f5ceae9662d759112352701c55a194ccdb78249e171ec2d97a604c9aba88b2b'
ARMS = ('static','norm','memory','blank','donor')
REFERENCES = ('frozen','JS')
CONTRASTS = [('memory',a) for a in ('norm','static','blank','donor',*REFERENCES)] + [('norm','static')]
PEOPLE = tuple(f'P{i:02d}' for i in range(1,11))
ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def source_manifest(path):
    lines = path.read_text().splitlines()
    require(len(lines)>0,'empty source manifest')
    for line in lines:
        digest,name=line.split(maxsplit=1)
        require(sha(path.parent/name)==digest,'source manifest mismatch '+name)
    return sha(path)


def controls_and_age(meta):
    require(sha(CONTROL)==CONTROL_SHA,'frozen controls changed')
    size=len(meta['subject'])
    with np.load(CONTROL) as z:
        require(np.array_equal(meta['subject'],z['subject']),'control row owners differ')
        out = {a:np.stack([z[a+'_'+o] for o in ORDERS]) for a in REFERENCES}
    perm_path=BASE/'runs/shuffle_perms.npz'
    require(sha(perm_path)==ORDER_SHA,'order archive changed')
    ages=np.empty((len(ORDERS),size),dtype=int)
    with np.load(perm_path) as data:
        for oi,o in enumerate(ORDERS):
            order=data[f'{o[:-1]}|{o[-1]}|all_loso']
            require(np.array_equal(np.sort(order),np.arange(size)),o+' permutation invalid')
            for s in np.unique(meta['stream']):
                rows=order[meta['stream'][order]==s]
                ages[oi,rows]=np.arange(len(rows))
    return out,ages


def folds(root, source_hash, people):
    expected={f'{a}|{o}' for a in ARMS for o in ORDERS}|{'index'}
    banks={a:np.full((len(ORDERS),len(people)),-1,dtype=np.int8) for a in ARMS}
    proofs={}
    for p in PEOPLE:
        row=np.flatnonzero(people==p)
        train=np.flatnonzero(people!=p)
        path=root/(p+'.npz')
        proof=root/(p+'.json')
        m=json.loads(proof.read_text())
        require(m['held']==p and set(m['train_people'])==set(PEOPLE)-{p},'producer source roster '+p)
        require(m['source_sha256']==source_hash and m['order_sha256']==ORDER_SHA and
                m['outputs_sha256']==sha(path) and
                m['train_rows_sha256']==hashlib.sha256(train.tobytes()).hexdigest(),'producer pin '+p)
        guard=m['guard']
        require(m['seconds']>0 and isinstance(guard,dict) and
                all(guard.get(k) is True for k in ('prefix','reset','step_equivalence','active','finite')),
                'missing or failed producer guard '+p)
        with np.load(path,allow_pickle=False) as z:
            require(set(z.files)==expected and np.array_equal(z['index'],row),'held row inventory '+p)
            for a in ARMS:
                for oi,o in enumerate(ORDERS):
                    values=z[f'{a}|{o}']
                    require(values.shape==(len(row),) and values.dtype.kind in 'iu'
                            and np.isin(values,(0,1,2)).all(),'invalid decision bank '+p+a+o)
                    banks[a][oi,row]=values
        proofs[p]=dict(proof_sha256=sha(proof),archive_sha256=m['outputs_sha256'],seconds=m['seconds'],
                       guard=m['guard'])
    require(all((v>=0).all() for v in banks.values()),'incomplete all-person matrix')
    return banks,proofs


def full_cells(labels,cells,active):
    # Conditional comparisons use exactly the same cells in all orders.
    all_cells=np.unique(cells)
    complete=[]
    for mask in active:
        complete.append({c for c in all_cells if set(labels[(cells==c)&mask])=={0,1,2}})
    common=sorted(set.intersection(*complete))
    return common,dict(total=len(all_cells),all_order_complete=len(common),
                       complete_by_order=dict(zip(ORDERS,map(len,complete))),common=common)


def summarize(metric,scorer,meta,labels,people,banks,active,name):
    cells=np.char.add(np.char.add(meta['stream'],'|'),meta['interface'])
    common,coverage=full_cells(labels,cells,active)
    if name=='all':
        require(len(common)==28 and set(people)==set(PEOPLE),'full metric coverage differs')
    if not common:
        return dict(coverage=coverage,status='no common complete cells')
    common_rows=np.isin(cells,common)
    cohort=[p for p in PEOPLE if np.any(common_rows & (people==p))]
    values={a:np.zeros((len(ORDERS),len(cohort))) for a in banks}
    counts=np.array([len(np.unique(cells[common_rows & (people==p)])) for p in cohort])
    for oi,mask in enumerate(active):
        for pi,p in enumerate(cohort):
            case=mask & common_rows & (people==p)
            for a,pred in banks.items():
                score=metric(labels[case],pred[oi,case],meta,case)
                require(score['n_cells']==counts[pi],'unpaired official cells')
                values[a][oi,pi]=100*score['balanced_accuracy_cell_mean']
    scores={}
    for a,table in values.items():
        absolute=np.mean([table[:4].mean(0),table[4:6].mean(0),table[6:].mean(0)],0)
        scores[a]=float(np.average(absolute,weights=counts))
    if name=='all':
        for a,expected in [('frozen',76.27933502145761),('JS',76.55173151211109)]:
            require(abs(scores[a]-expected)<1e-7,'historical baseline mismatch '+a)
    contrasts={}
    for a,b in CONTRASTS:
        delta=values[a]-values[b]
        contrasts[a+'-'+b]=dict(official=scorer.cell_weighted_summary(delta,ORDERS,counts),
                                  people_order=scorer.summarize(delta,ORDERS,cohort))
    return dict(coverage=coverage,people=cohort,cell_counts=counts.tolist(),scores=scores,contrasts=contrasts)


def main(root):
    src=source_manifest(root/'source.sha256')
    file=BASE/'data/proteus_windows.npz'
    require(sha(file)==DATA_SHA,'dataset changed')
    require(sha(METRICS/'source.sha256')=='9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd',
            'metric source changed')
    source_manifest(METRICS/'source.sha256')
    sys.path.insert(0,str(METRICS))
    from measure import official
    metric=official()
    path=BASE/'studies/14nov/newmember/score_s_v66p.py'
    require(sha(path)=='ae387cd5a7b116fc1259001d4fe449ba1065b0cb5943c117d4f65ec87a0b40d4',
            'interval scorer changed')
    spec=importlib.util.spec_from_file_location('paired379',path)
    scorer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    require(tuple(scorer.ORDERS)==ORDERS and tuple(scorer.PEOPLE)==PEOPLE,'canonical order differs')
    with np.load(file) as z:
        meta={k:z[k] for k in ('subject','session','interface')}
    meta['stream']=np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    people=np.array([s.split('/')[-1] for s in meta['subject']])
    banks,ages=controls_and_age(meta)
    produced,proofs=folds(root,src,people)
    banks.update(produced)
    before=dict(source_sha=src,controls_sha=CONTROL_SHA,orders_sha=ORDER_SHA,folds=proofs)
    (root/'before_score379.json').write_text(json.dumps(before,indent=2))
    with np.load(file) as z:
        labels=z['y']
    output={name:summarize(metric,scorer,meta,labels,people,banks,mask,name) for name,mask in [
            ('all',ages>=0),('age0_31',ages<32),('age32_plus',ages>=32)]}
    output['before_score_sha256']=sha(root/'before_score379.json')
    (root/'result379.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output['all']['scores']),flush=True)


if __name__=='__main__':
    main(Path.cwd())
