"""Export pinned source-only native linear reference; never read held labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp,softmax
from sklearn.preprocessing import StandardScaler

BASE=Path('/mnt/lustre/work/gehler/gfh527')
MATRIX=BASE/'ni26_s311_cpu_kernel_v1_20261116/runs/s311_smoke/source_matrix.npz'
FIXTURE=BASE/'ni26_s301_native_reference_20261116/runs/s301_reference'
HEAD=BASE/'ni26_solver343_baseline_v1_20261117/cold_sklearn_1.npz'
REFERENCE=BASE/'ni26_s301_native_reference_all_20261116/runs/s301_reference_all/P01.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if sha(MATRIX)!='a38ec018aaf2f727916fa402bb082659543dcfbf758d44e6ee3f9819e4ef8b24':raise ValueError('source matrix changed')
    with np.load(MATRIX,allow_pickle=False) as z:x,y=z['x'].copy(),z['y'].copy()
    with np.load(HEAD,allow_pickle=False) as z:theta=z['theta'].copy()
    with np.load(REFERENCE,allow_pickle=False) as z:mean,scale,rows=z['mean'].copy(),z['scale'].copy(),z['fit_rows'].copy()
    if x.shape!=(2465,44239) or not np.array_equal(np.unique(y),[0,1,2]):raise ValueError('source shape/labels')
    logits=x@theta[:,:-1].T+theta[:,-1]
    objective=float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])+np.square(theta[:,:-1]).sum()/(2*.3*len(y)))
    scaler_x=x[:96,:257].copy();scaler_x[:,0]=scaler_x[0,0]
    scaler=StandardScaler().fit(scaler_x)
    arrays=dict(x=x,y=y,C=np.array(.3),threads=np.array(4,dtype=np.int64),reference_nit=np.array(82,dtype=np.int64),
                reference_nfev=np.array(88,dtype=np.int64),reference_objective=np.array(objective),
                source_probabilities=softmax(logits,axis=1),scaler_x=scaler_x,scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_,scaler_expected=scaler.transform(scaler_x))
    manifest=json.loads((FIXTURE/'manifest.json').read_text());pins={}
    for order in ('within0','full0'):
        path=FIXTURE/f'P01_{order}.npz';pins[order]=sha(path)
        if pins[order]!=manifest['files'][path.name]:raise ValueError('query fixture changed')
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['mean'],mean) or not np.array_equal(z['scale'],scale) or not np.array_equal(z['fit_rows'],rows):raise ValueError('actual source scaler/exclusion differs')
            query=np.ascontiguousarray((z['expected_features']-mean)/scale)
        arrays[order+'_x']=query;arrays[order+'_probabilities']=softmax(query@theta[:,:-1].T+theta[:,-1],axis=1)
    output={k:torch.from_numpy(np.ascontiguousarray(v)) if np.ndim(v) else torch.as_tensor(v) for k,v in arrays.items()}
    output['native_format']=torch.tensor(1);torch.save(output,args.output)
    proof=dict(source_matrix_sha256=sha(MATRIX),reference_head_sha256=sha(HEAD),source_reference_sha256=sha(REFERENCE),
               query_fixtures=pins,no_held_labels_or_accuracy=True,fixture_sha256=sha(args.output),
               source_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),source_shape=list(x.shape),
               source_export_only=True,threads=4,scaler_constant_column=0)
    args.output.with_suffix('.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
