"""Full real-data preparation, source fit and numerical guards; no query labels/scores."""
import json
import os
import time
from pathlib import Path
import numpy as np
from model import fit,transform
from guards import cache_probe,check
from inference import banks,density_stream
from data import metadata,labels,sha,official
from feature_io import load


def main():
    root=Path(os.environ['GEN_ROOT']);start=time.monotonic()
    for line in (root/'source.sha256').read_text().splitlines():
        pin,name=line.split(maxsplit=1)
        if sha(root/name)!=pin:raise ValueError('sourcechanged')
    bank,record,manifest=load('source7');d,_,_,_,perms=metadata()
    x,rows,source,query,people=[bank[k] for k in ('features','rows','source','query','people')]
    preparation=time.monotonic()-start;probe_result={'dependency_manifest_sha256':manifest}
    metric=official();y=labels(rows[source],len(d['subject']))
    t=time.monotonic();model=fit(x[source],y,people[source]);fit_s=time.monotonic()-t
    values=x[source[:17]];proof=check(model,values,1.)
    t=time.monotonic();z=transform(model,x[source]);banked=banks(model,z,1.);bank_s=time.monotonic()-t
    t=time.monotonic()
    for arm,v in banked.items():density_stream(model,v,1.,arm)
    density_s=time.monotonic()-t
    # Sevenheld-source mixture episodes×3precisions×8orders, then6finalarms×8orders.
    # Held-source bank sizes sum to onefullsource bank; finalqueries add~20%.
    projected=1.5*(8*fit_s+2*bank_s+6*density_s+8*preparation+60)
    fixed=json.loads((Path(os.environ['FIXED_ROOT'])/'manifest.json').read_text())
    spent_fixed=float(fixed['seconds'])*2+378+60  # twoallocatedCPU, prior tinyallocations, startupreserve
    total_core=spent_fixed+(time.monotonic()-start)*6+projected*6
    result=dict(passed=True,labels_opened='P04-P10 source only',query_labels_opened=False,P01_used=False,
                source_sha256=sha(root/'source.sha256'),feature_sha256=record['file_sha256'],
                cache_probe=probe_result,proof=proof,source_rows=rows[source].tolist(),query_rows=rows[query].tolist(),
                preparation_seconds=preparation,fit_seconds=fit_s,full_source_bank_seconds=bank_s,full_source_density_seconds=density_s,
                projected_remaining_pilot_seconds=projected,projected_cpu_coreseconds_including_preparation_and_smoke=total_core,
                cost_pass=total_core<=7200,metric_path=metric.__file__,total_seconds=time.monotonic()-start)
    with (root/'smoke.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
