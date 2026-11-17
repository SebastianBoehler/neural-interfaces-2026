"""Measure the failed exact inner guard; never relax it or release downstream work."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from package_guard import load,instrument,frame
from proofs import ACTIVE,manifest
from reference import sha


def main():
    ws=Path.cwd();manifest(ws)
    active=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_active3')
    manifest(active);torch.set_num_threads(1)
    with np.load(active/'data/proteus_windows.npz',allow_pickle=False) as z:
        rows=np.flatnonzero(z['subject']=='Dreyer2026Proteus/P01')[:2];x=z['X'][rows].copy();channels=z['ch_names']
    models={k:instrument(load(active/p,k,channels)) for k,p in
            (('baseA','draft_base'),('baseB','draft_base'),('off','draft_off'),('on','draft_jsall'))}
    records=[]
    for t,value in enumerate(x):
        traces={k:frame(m,value) for k,m in models.items()};reference=traces['baseA'];row=dict(t=t,comparisons={})
        for key in ('baseB','off','on'):
            item=traces[key];fields={}
            for field in ('mix','core1','u','shadow','out_v5'):
                a=np.asarray(reference[2][field]);b=np.asarray(item[2][field])
                fields[field]=dict(exact=np.array_equal(a,b),max_error=float(np.abs(a-b).max()),a=a.tolist(),b=b.tolist())
            row['comparisons'][key]=dict(final_exact=np.array_equal(reference[1],item[1]),fields=fields)
        records.append(row)
    report=dict(status='failed_guard_diagnostic_only',records=records,source_sha256=sha(ws/'source.sha256'),
                active_source_sha256=sha(active/'source.sha256'),release_allowed=False,accuracy_computed=False)
    (ws/'diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({str(r['t']):{k:{f:(v['exact'],v['max_error']) for f,v in c['fields'].items()} for k,c in r['comparisons'].items()} for r in records}),flush=True)


if __name__=='__main__':main()
