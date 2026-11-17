"""Original-bank identity first, then three matched frozen-statistic counterfactuals."""
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from source349 import load,STUDY,permutations,ORDERS,bn,install,parameter_sha,state_sha,sha,digest

PREP=Path('/mnt/lustre/work/gehler/gfh527/ni26_recal349_source_v1_20261117')


@torch.no_grad()
def predict(model,d,rows,perm):
    before=state_sha(model.net);parameters=parameter_sha(model.net);rank=np.empty(len(perm),int);rank[perm]=np.arange(len(perm));p=np.empty((len(rows),3));maximum=0.
    for stream in dict.fromkeys(d['stream'][rows]):
        loc=np.flatnonzero(d['stream'][rows]==stream);loc=loc[np.argsort(rank[rows[loc]],kind='stable')];rr=rows[loc];out=model.predict_proba_stream(rr)
        n=min(7,len(rr));prefix=model.predict_proba_stream(rr[:n]);error=float(np.max(np.abs(out[:n]-prefix)));maximum=max(maximum,error)
        if error>1e-4:raise ValueError('native source-frozen query prefix failed')
        other=rows[d['stream'][rows]!=stream]
        if len(other):model.predict_proba_stream(other[:n])
        repeated=model.predict_proba_stream(rr[:n]);net=copy.deepcopy(model.net).eval()
        copied=torch.softmax(net(torch.from_numpy(model._aligned(rr[:n],True))).double(),1).numpy()
        if not np.array_equal(prefix,repeated) or not np.array_equal(prefix,copied):raise ValueError('native interleave/reset/net-only deepcopy failed')
        p[loc]=out
    if before!=state_sha(model.net) or parameters!=parameter_sha(model.net) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6):raise ValueError('query mutated frozen state/nonfinite')
    return p,dict(prefix_maxabs=maximum,state_sha256=before,parameter_sha256=parameters,permutation_sha256=digest(perm))


def main():
    root=Path(os.environ['RECAL349_ROOT']);start=time.monotonic();torch.set_num_threads(6)
    prep=json.loads((PREP/'prepare.json').read_text())
    if not prep['passed'] or not prep['budget_passed'] or prep['source_sha256']!=sha(PREP/'source.sha256') or sha(PREP/'buffers.npz')!=prep['buffer_sha256']:raise ValueError('complete source/oracle/cost/statistics proof required')
    for line in (PREP/'source.sha256').read_text().splitlines():
        pin,name=line.split(maxsplit=1)
        if sha(PREP/name)!=pin:raise ValueError('prepared source diagnostic/statistics code changed:'+name)
    for name in ('source349.py','moments349.py'):
        if sha(root/name)!=sha(PREP/name):raise ValueError('source/statistic algorithm changed after smoke')
    d,fit,people,_,_,_,models,records,_=load();query=np.flatnonzero(np.isin(people,['P02','P03','P09']));perms=permutations(len(people))
    with np.load(PREP/'buffers.npz') as z:buffers={k:z[k].copy() for k in z.files}
    oldproof=json.loads((STUDY/'before_labels.json').read_text())
    if sha(STUDY/'complete_predictions.npz')!=oldproof['predictions_sha256']:raise ValueError('original347complete bank changed')
    with np.load(STUDY/'complete_predictions.npz') as z:
        if not np.array_equal(z['index'],query):raise ValueError('original347query rows changed')
        originals={a:{o:z[a+'_'+o].copy() for o in ORDERS} for a in ('A','B','C')}
    banks={'A':originals['A']};guards={};identity={};states={a:state_sha(m.net) for a,m in models.items()}
    # ALL original banks must pass before any recalibration is installed.
    for arm,model in models.items():
        banks[arm]={};guards[arm]={};identity[arm]={}
        for order,perm in perms.items():
            p,check=predict(model,d,query,perm);error=float(np.max(np.abs(p-originals[arm][order])))
            if error>1e-6 or not np.array_equal(p.argmax(1),originals[arm][order].argmax(1)):raise ValueError('original347ALL8posterior/label identity failed')
            banks[arm][order]=p;guards[arm][order]=check;identity[arm][order]=error
    (root/'original_before_recal.json').write_text(json.dumps(dict(passed=True,identity=identity,source_sha256=sha(root/'source.sha256'),query_labels_opened=False)))
    changed={}
    for name,arm in (('B_recal','B'),('C_mixed_recal','C'),('C_official_recal','C')):
        model=models[arm];changed[name]=install(model,buffers[name+'_mean'],buffers[name+'_var'])
        expected=prep['records'][arm]['variants'][name]['changed']
        if changed[name]!=expected:raise ValueError('source-only changed-buffer/parameter identity differs')
        banks[name]={};guards[name]={}
        for order,perm in perms.items():banks[name][order],guards[name][order]=predict(model,d,query,perm)
        effect=max(float(np.max(np.abs(banks[name][o]-banks[arm][o]))) for o in ORDERS)
        if effect==0:raise ValueError('changed source BN buffers have exactly no prediction effect')
        changed[name]['posterior_maxabs_vs_original']=effect
        with torch.no_grad():bn(model).running_mean.copy_(torch.from_numpy(buffers[arm+'_old_mean']));bn(model).running_var.copy_(torch.from_numpy(buffers[arm+'_old_var']))
        if state_sha(model.net)!=states[arm]:raise ValueError('originalmodel restore failed')
    np.savez(root/'predictions.npz',index=query,**{a+'_'+o:p for a,bb in banks.items() for o,p in bb.items()})
    proof=dict(passed=True,all6arms_all3people_all8orders=True,source_sha256=sha(root/'source.sha256'),
        prepare_sha256=sha(PREP/'prepare.json'),buffer_sha256=prep['buffer_sha256'],predictions_sha256=sha(root/'predictions.npz'),
        original_before_recal_sha256=sha(root/'original_before_recal.json'),identity=identity,guards=guards,changed=changed,
        source_rows_sha256=digest(fit),query_rows_sha256=digest(query),query_labels_opened=False,optimizer_steps=0,seconds=time.monotonic()-start)
    (root/'prediction_proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k not in ('guards','changed')}),flush=True)


if __name__=='__main__':main()
