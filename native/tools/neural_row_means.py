"""Append original NumPy row-mean oracles; preserve every existing source tensor."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    if args.output.exists():raise ValueError('refusing to overwrite source oracle')
    source=torch.load(args.source,weights_only=True,map_location='cpu');derived=dict(source)
    for mode,count in [('known',8),('unknown',31)]:
        x=source[mode+'_x']
        if x.dtype!=torch.float32 or x.shape!=(count,41,480):raise ValueError('fixed existing source scope changed')
        derived[mode+'_row_means']=torch.from_numpy(x.numpy().mean(-1,keepdims=True))
    torch.save(derived,args.output);check=torch.load(args.output,weights_only=True,map_location='cpu')
    for key,value in source.items():
        other=check[key]
        if value.dtype!=other.dtype or value.shape!=other.shape or value.numpy().tobytes()!=other.numpy().tobytes():raise ValueError('original fixture tensor changed:'+key)
    print(json.dumps(dict(original_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),derived_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        preserved_original_keys=len(source),added_mean_values=1599,numpy_version=np.__version__,neural_forwards=0,optimizer_steps=0,task_labels_loaded=False)),flush=True)


if __name__=='__main__':main()
