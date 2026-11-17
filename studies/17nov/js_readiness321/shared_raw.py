"""Actual optimized factory: six held raw streams versus full traces and historical archives."""
import json
from pathlib import Path

import numpy as np

from reference import SOURCE,sha

REPAIR=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_inputrepair1')
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def verify_raw(g,device):
    recovery=json.loads((REPAIR/'recovery.json').read_text())
    data=SOURCE/'data/proteus_windows.npz'
    with np.load(data,allow_pickle=False) as f:d={k:f[k] for k in ('X','subject','session')}
    d['stream']=np.char.add(np.char.add(d['subject'],'|'),d['session'])
    data_sha=sha(data)
    if recovery.get('raw_input_all_seeds_orders_passed') is not True:raise ValueError('original raw fidelity missing')
    with np.load(REPAIR/'held_input.npz',allow_pickle=False) as z:p={k:z[k].copy(order='K') for k in z.files}
    if float(p['k'])!=5.:raise ValueError('actual source input prior must be5;latent BN2 stays inside traces')
    files=[];archives=[]
    for row in recovery['models']:
        if sha(row['trace'])!=row['trace_sha256']:raise ValueError('held source trace changed')
        files.append(Path(row['trace']).name)
        archive=SOURCE/f"runs/coral304/conditional30_s{row['seed']}_P01.npz"
        if sha(archive)!=row['archive_sha256']:raise ValueError('historical raw archive changed')
        a=np.load(archive,allow_pickle=False)
        if json.loads(str(a['metadata']))['data_sha256']!=data_sha:raise ValueError('raw tensor identity differs')
        archives.append(a)
    z={'js_cc_files':np.array(files),'js_seeds':np.arange(6),**{'js_'+k:v for k,v in p.items()}};results={}
    for order in ORDERS:
        index=archives[0][order+'|index']
        if any(not np.array_equal(index,a[order+'|index']) for a in archives):raise ValueError('source rows differ')
        bank=g['build_js_bank'](z,device,REPAIR/'held')
        full=g['NetBank']([dict(p,sjit_file=np.array(f)) for f in files],np.ones(6,bool),device,REPAIR/'held')
        got=np.empty((6,len(index),3));maximum=0.
        for session in dict.fromkeys(d['stream'][index]):
            loc=np.flatnonzero(d['stream'][index]==session);bank.reset_state();full.reset_state()
            for i in loc:
                x=d['X'][index[i]].astype(np.float32);a=np.stack([p for _,p in bank(x)]);b=np.stack([p for _,p in full(x)])
                maximum=max(maximum,float(abs(a-b).max()));got[:,i]=a
        historical=max(float(abs(got[s]-archives[s][order+'|proba']).max()) for s in range(6))
        if maximum>1e-8 or historical>=1e-4:raise ValueError('optimized raw fidelity failed')
        results[order]=dict(full_trace_max_error=maximum,historical_max_error=historical,rows=len(index),seeds=6)
    return results
