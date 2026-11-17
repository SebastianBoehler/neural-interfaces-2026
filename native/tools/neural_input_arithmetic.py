"""Inspect existing SOURCE input arithmetic; no neural forward or optimizer."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.jit._pickle import restore_type_tag


def difference(a,b):
    a=np.asarray(a);b=np.asarray(b);delta=np.abs(a-b)
    return dict(max_abs=float(delta.max()),different=int(np.count_nonzero(a!=b)),rms=float(np.sqrt(np.mean(delta.astype(np.float64)**2))))


def main():
    parser=argparse.ArgumentParser()
    for key in ('package','fixture','native','project','out'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    sys.path.insert(0,str(args.project/'2026-competition/tracks/bci_decoding'))
    spec=importlib.util.spec_from_file_location('exact_source_input_arithmetic',args.package/'submission.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with np.load(args.package/'weights.npz',allow_pickle=False) as f:z={key:f[key] for key in f.files}
    f=torch.load(args.fixture,weights_only=True,map_location='cpu')
    with torch.serialization.safe_globals([restore_type_tag]):native=torch.load(args.native,weights_only=True,map_location='cpu')
    records=[]
    for j,member in enumerate(f['input_ids'].tolist()):
        p=module.with_prefix(z,f'net{member}_')
        if not bool(p['align']):continue
        reference=module.InputStream(p);prior=torch.from_numpy(np.array(p['G'],np.float64,copy=True,order='C'))
        sums={name:torch.zeros_like(prior) for name in ('torch_mean','original_mean')}
        for window,x in enumerate(f['known_x']):
            raw=x.numpy();original_mean=raw.mean(-1,keepdims=True);torch_mean=x.mean(-1,True)
            expected=reference(raw);same=f['known_input'][window,j].numpy()
            stages={}
            for name,mean in [('torch_mean',torch_mean),('original_mean',torch.from_numpy(original_mean))]:
                centered=(x-mean).to(torch.float64);cov=centered@centered.T/480
                shrink=float(p['shrink']);cov=(1-shrink)*cov+shrink*cov.trace()/41*torch.eye(41,dtype=torch.float64)
                sums[name]=sums[name]+cov;k=float(p['k']);R=(k*prior+sums[name])/(k+window+1)
                values,vectors=torch.linalg.eigh(R,UPLO='L');P=(vectors*values.clamp_min(1e-12).rsqrt()[None,:])@vectors.T
                transformed=(P@x.to(torch.float64)).to(torch.float32)
                stage=dict(output=difference(transformed.numpy(),expected),covsum=difference(sums[name].numpy(),reference.cov_sum))
                if window==0:stage['native_capture']=difference(transformed.numpy(),native[f'input_{member}'].numpy())
                stages[name]=stage
            records.append(dict(member=member,window=window,original_vs_golden=difference(expected,same),mean=difference(torch_mean.numpy(),original_mean),stages=stages))
    args.out.write_text(json.dumps(dict(torch_version=torch.__version__,numpy_version=np.__version__,source_windows=8,neural_forwards=0,optimizer_steps=0,
        task_labels_loaded=False,accuracy_computed=False,tolerance_changed=False,records=records),indent=2))


if __name__=='__main__':main()
