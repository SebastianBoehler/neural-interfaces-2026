"""Post-fit numerical check at genuinely input-dependent learned transitions, no score."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_oscillatory331_v2_20261117')
sys.path.insert(0,str(ROOT))
from data331 import load,labels,sha
from train331 import make
from oracle331 import compare


def main():
    torch.set_num_threads(6)
    if sha(ROOT/'results.json')!='863b18d18248042ac020013d8234a9dc3aad8b19741f55d10c737a1e032fe972':raise ValueError('complete report changed')
    report=json.loads((ROOT/'results.json').read_text());d,fit,query,people=load()
    with np.load(ROOT/'scaler.npz') as z:
        rms=float(z['rms']);mean=torch.from_numpy(z['mean']);sd=torch.from_numpy(z['sd'])
    rows=np.flatnonzero(people=='P04')[:1]
    x=torch.from_numpy((d['X'][rows]/rms).astype(np.float32));y=torch.from_numpy(labels(rows,len(d['X'])))
    checkpoint=ROOT/'S.pt'
    if sha(checkpoint)!=report['checkpoints']['S']:raise ValueError('learned checkpoint changed')
    model=make('S',mean,sd);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state'],strict=True)
    with torch.no_grad():
        z=model.stem(x-x.mean(1,keepdim=True)).transpose(1,2);variation=[]
        for layer in model.layers:
            ar,ai,_,_=layer.coefficients(z);rho=(ar.square()+ai.square()).sqrt()
            variation.append(float((rho.max(1).values-rho.min(1).values).max()));z=layer(z)
    if min(variation)<=1e-6:raise ValueError('learned transition not genuinely input-dependent')
    result=dict(checkpoint_sha256=sha(checkpoint),source_report_sha256=sha(ROOT/'results.json'),
        source_rows=rows.tolist(),query_labels_opened=False,transition_temporal_variation=variation,
        oracle={str(dtype):compare(model,x,y,dtype) for dtype in (torch.float64,torch.float32)})
    path=Path(__file__).resolve().parent/'results.json'
    with path.open('x') as out:json.dump(result,out,indent=2)
    print('OK: learned selective outputs/gradients match independent complex recurrence; no query score',flush=True)


if __name__=='__main__':main()
