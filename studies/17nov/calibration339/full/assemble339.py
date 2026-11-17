"""Complete twenty nested banks BEFORE temperatures; all T BEFORE any outer metrics."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
PROD=Path('/mnt/lustre/work/gehler/gfh527/ni26_calibration339_full_v2_20261117')
PROTO=Path('/mnt/lustre/work/gehler/gfh527/ni26_calibration339_parallel_v1_20261117')
sys.path.insert(0,str(PROTO))
from shared339 import src,BASE_SHA,sha,digest,ensemble,apply,weights,fit
MOMENT=Path('/mnt/lustre/work/gehler/gfh527/ni26_moments332_v2_20261117')
ARMS=('RF','RB','RV','CF','CB','CV');VIEWS=('F','B','V')
SOURCE='a4b9b0c34e30712cbc395f89ceff6e5db6bc8ffb5d128453b511bba199a8eca3'


def main():
    root=Path(os.environ['ASSEMBLY_ROOT']);source=sha(PROD/'source.sha256')
    if source!=SOURCE:raise ValueError('produceridentity')
    src.check()
    with np.load(src.data323.DATA) as z:d={k:z[k] for k in ('X','subject','session','interface')}
    people=np.array([str(s).split('/')[-1] for s in d['subject']]);d['stream']=np.char.add(np.char.add(d['subject'],'|'),d['session'])
    records={};banks={};inputs={};kernel=None
    # Complete immutable332 variances and323 endpoints also verified BEFOREcalibrationlabels.
    moment_report=MOMENT/'results.json'
    if sha(moment_report)!='ec51316c7fc474d8960961e93783177336b4a21c656e6a62ede5571d5d3b65ee':raise ValueError('complete332reportchanged')
    mr=json.loads(moment_report.read_text());inputs[str(moment_report)]=sha(moment_report)
    for i,outer in enumerate(src.PEOPLE):
        cal=src.PEOPLE[(i+1)%10];rows=np.flatnonzero(~np.isin(people,[outer,cal]));q=np.flatnonzero(people==cal)
        for seed in (0,1):
            folder=(PROTO/f'seed{seed}') if outer=='P01' else (PROD/outer/f'seed{seed}')
            m=json.loads((folder/'record.json').read_text());expected_source=sha(PROTO/'source.sha256') if outer=='P01' else SOURCE
            if (m['seed'],m['outer'],m['cal'],m['source_sha256'],m['scientific_base'])!=(seed,outer,cal,expected_source,BASE_SHA):raise ValueError('nestedproduceridentity')
            if m['calibration_labels_opened'] or m['outer_labels_opened'] or m['fit_people']!=sorted(set(people[rows])):raise ValueError('nestedsourceexclusion')
            for name,array in (('fit_rows',rows),('cal_rows',q),('fit_input',d['X'][rows])):
                if m[name+'_sha256']!=digest(array):raise ValueError('nestedrowschanged')
            for name,key in (('model.pt','checkpoint_sha256'),('logits.npz','logits_sha256')):
                p=folder/name
                if sha(p)!=m[key]:raise ValueError('immutablebank/checkpoint')
                inputs[str(p)]=sha(p)
            inputs[str(folder/'record.json')]=sha(folder/'record.json')
            src.model322.SEED=seed;groups=src.model322.batches(d['stream'][rows]);batch=hashlib.sha256(b''.join(b.tobytes() for ep in groups for b in ep)).hexdigest()
            if m['training']['steps']!=sum(map(len,groups)) or m['batches_sha256']!=batch or m['training']['max_gradient']<=0:raise ValueError('native40epochschedule')
            if kernel is None:kernel=m['kernel']
            if m['kernel']!=kernel or m['device']!='NVIDIA A100-PCIE-40GB':raise ValueError('kernel/deviceflags')
            src.verify(outer,seed)
            with np.load(src.PROD/'output'/f'{outer}_s{seed}'/'result.npz') as z:old=json.loads(str(z['metadata']))
            if m['initial_param_sha256']!=old['initial_sha256']['P0'] or m['original_source9_checkpoint']!=old['checkpoints']['P0']:raise ValueError('originalseed/checkpoint')
            keys={kind+'|'+v+'|'+o for kind in ('held8','insample9') for v in VIEWS for o in src.ORDERS}
            with np.load(folder/'logits.npz') as z:
                if set(z.files)!={'index'}|keys or not np.array_equal(z['index'],q):raise ValueError('complete3view8orderbank')
                b={v+'|'+o:z['held8|'+v+'|'+o].copy() for v in VIEWS for o in src.ORDERS}
                if any(x.shape!=(len(q),3) or not np.isfinite(x).all() for x in b.values()):raise ValueError('invalidsourceheldlogits')
                banks[outer,seed]=b
            p=MOMENT/'output'/f'{outer}_s{seed}.npz'
            if sha(p)!=mr['hashes'][p.name]:raise ValueError('332foldchanged')
            inputs[str(p)]=sha(p);records[f'{outer}_s{seed}']=m
    # ALL20fit/bank/checkpoint/source/seed/kernel/schedule guards complete. Now fit30 SOURCEheld T.
    temperatures={}
    for i,outer in enumerate(src.PEOPLE):
        cal=src.PEOPLE[(i+1)%10];q=np.flatnonzero(people==cal)
        y=src.source_labels.read_training_labels(src.data323.DATA,q,len(people));cells=np.char.add(np.char.add(d['stream'][q],'|'),d['interface'][q]);w=weights(y,cells)
        temperatures[outer]={v:fit({o:ensemble([banks[outer,s][v+'|'+o] for s in (0,1)]) for o in src.ORDERS},y,w,src.ORDERS) for v in VIEWS}
    (root/'temperatures.json').write_text(json.dumps(dict(temperatures=temperatures,records=records,source=SOURCE,outer_metrics_computed=False),indent=2))
    # ALL30T frozen before assembling/applying OUTERmembers; no outer labels or scores here.
    outputs={a:{o:np.full((len(people),3),np.nan) for o in src.ORDERS} for a in ARMS};identity={}
    for outer in src.PEOPLE:
        q=np.flatnonzero(people==outer)
        for o in src.ORDERS:
            for v,oldname in (('F','P0F'),('B','P0C'),('V','PV')):
                values=[]
                for seed in (0,1):
                    p=(src.PROD/'output'/f'{outer}_s{seed}'/'result.npz') if v!='V' else (MOMENT/'output'/f'{outer}_s{seed}.npz')
                    with np.load(p) as z:
                        if not np.array_equal(z['index'],q):raise ValueError('outerbankroworder')
                        values.append(z[oldname+'|'+o].copy())
                    inputs[str(p)]=sha(p)
                raw=np.mean(values,axis=0)
                if not np.isfinite(raw).all() or (raw<=0).any() or not np.allclose(raw.sum(1),1,atol=1e-6):raise ValueError('outerrawproba')
                lp=np.log(raw);t=temperatures[outer][v]['T'];changed=np.exp(apply(lp,t))
                if not np.array_equal(raw.argmax(1),changed.argmax(1)):raise ValueError('outertemperatureargmax')
                outputs['R'+v][o][q]=raw;outputs['C'+v][o][q]=changed
    # RawF/B exact mean-stream reproduction; rawV exact332mean by construction, with hashes above.
    for v,oldname in (('F','n323p0f'),('B','n323p0c')):
        validation=json.loads((src.PROD/'runs/n323_streams/validation.json').read_text())
        for o in src.ORDERS:
            value=[]
            for s in (0,1):
                p=src.PROD/'runs/n323_streams'/f'ttt18{o}_{oldname}_base-all_loso-s{s}.npz'
                if sha(p)!=validation['streams'][p.name]:raise ValueError('323canonicalstreamchanged')
                with np.load(p) as z:value.append(z['proba'].copy())
                inputs[str(p)]=sha(p)
            if not np.array_equal(outputs['R'+v][o],np.mean(value,axis=0)):raise ValueError('rawFBnotbitwise323')
    dest=root/'streams';dest.mkdir(exist_ok=False);hashes={}
    for a in ARMS:
        for o,p in outputs[a].items():
            if not np.isfinite(p).all():raise ValueError('incompletematrix')
            family='n339'+a.lower();name=f'ttt18{o}_{family}_base-all_loso-ensemble2.npz'
            meta=dict(family=family,order=o,seeds=[0,1],operation='arithmetic_probability_mean' if a[0]=='R' else 'temperature_after_arithmetic_probability_mean',source=SOURCE,temperatures_sha256=sha(root/'temperatures.json'))
            with (dest/name).open('xb') as f:np.savez(f,index=np.arange(len(people)),proba=p,metadata=json.dumps(meta))
            hashes[name]=sha(dest/name)
    inputs[str(root/'temperatures.json')]=sha(root/'temperatures.json')
    (root/'validation.json').write_text(json.dumps(dict(streams=hashes,inputs=inputs,source=SOURCE,outer_metrics_computed=False,raw_FB_bitwise323=True,raw_V_hashverified332=True),indent=2))
    print('ALL20sourcebanks→ALL30temperatures→48outerstreams;NOoutermetrics',flush=True)


if __name__=='__main__':main()
