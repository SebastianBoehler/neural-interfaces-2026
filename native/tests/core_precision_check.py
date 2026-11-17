"""Report every unchanged guard before failing a native source component run."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.jit._pickle import restore_type_tag


def load(path):
    with torch.serialization.safe_globals([restore_type_tag]):return torch.load(path,weights_only=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['car','check']);parser.add_argument('reference',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--native',type=Path);args=parser.parse_args()
    reference=load(args.reference);torch.set_num_threads(1)
    if args.mode=='car':
        raw=reference['x'].numpy()
        if raw.shape!=(96,41,480) or raw.dtype!=np.float32:raise ValueError('exact original96 source windows required')
        car=np.stack([x-x.mean(0,keepdims=True) for x in raw])
        torch.save({'native_format':torch.tensor(1),'car':torch.from_numpy(car)},args.output)
        return
    native=load(args.native);results={}
    for key in ('p1','p2','features','tail','reapply'):
        a,b=native[key],reference[key]
        if a.shape!=b.shape or not torch.isfinite(a).all():raise ValueError('invalid source output '+key)
        error=float(abs(a-b).max());limit=1e-4 if key=='features' else 1e-7
        changed=None if key=='features' else int((a.argmax(-1)!=b.argmax(-1)).sum())
        results[key]=dict(max_abs=error,limit=limit,changed_labels=changed,passed=error<=limit and not changed)
        print(key,json.dumps(results[key]),flush=True)
    passed=all(r['passed'] for r in results.values())
    args.output.write_text(json.dumps(dict(passed=passed,results=results,source_windows=int(reference['x'].shape[0]),
                          no_labels_fits_or_accuracy=True,host_python_fixture=True),indent=2))
    if not passed:raise ValueError('unchanged native component guards failed')


if __name__=='__main__':main()
