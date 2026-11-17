"""Source-only numeric diagnosis of the first original-prefix failure, no bound change."""
import importlib.util
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from metadata347 import source,plan,restored,sha,digest
from prepare347 import OWNER,OWNER_SHA


def main():
    root=Path(os.environ['CROP_ROOT']);start=time.monotonic();torch.set_num_threads(2)
    d,fit,held,people,y=source();groups,_,_=plan(d,fit,people,y);g=groups[0]
    if sha(OWNER)!=OWNER_SHA:raise ValueError('raw owner changed')
    spec=importlib.util.spec_from_file_location('raw347diagnosticowner',OWNER);owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    model,_=restored(d,fit,held);values=owner.processed(Path(g['path']),d['ch_names'].astype(str).tolist())
    n=7;rows=np.asarray([r['row'] for r in g['rows'][:n]],np.int64)
    uv=np.stack([values[:,r['original_start']:r['original_start']+480] for r in g['rows'][:n]])
    inputs={'cached':d['X'][rows], 'old_doubleV_to_uV_then_float32':uv.astype(np.float32),
        'official_float32V_then_scale':(uv/1e6).astype(np.float32)*np.float32(1e6)}
    expected=model._aligned(rows,True);state=__import__('common346').state_sha(model.net);outputs={};report={}
    for name,value in inputs.items():
        x=(value.astype(np.float64)/model.scale).astype(np.float32)
        with torch.no_grad():outputs[name]=torch.softmax(model.net(torch.from_numpy(x)).double(),1).numpy()
        report[name]=dict(raw_maxabs=float(np.max(np.abs(value-d['X'][rows]))),
            aligned_maxabs=float(np.max(np.abs(x-expected))),raw_bitwise=bool(np.array_equal(value,d['X'][rows])),
            input_sha256=digest(value))
    with torch.no_grad():actual=torch.softmax(model.net(torch.from_numpy(expected)).double(),1).numpy();repeat=torch.softmax(model.net(torch.from_numpy(expected)).double(),1).numpy()
    for name,p in outputs.items():report[name].update(probability_maxabs=float(np.max(np.abs(p-actual))),changed_labels=int(np.sum(p.argmax(1)!=actual.argmax(1))))
    result=dict(source_sha256=sha(root/'source.sha256'),source_file=g['path'],rows_sha256=digest(rows),diagnostics=report,
        cached_repeat_error=float(np.max(np.abs(actual-repeat))),native_align=model.align,native_scale=model.scale,
        net_training=model.net.training,BN_k=[m.k for m in model.sbns()],state_unchanged=state==__import__('common346').state_sha(model.net),
        extractor_source_sha256=sha(__import__('inspect').getfile(owner.EegExtractor)),
        held_transformed=False,held_labels_opened=False,seconds=time.monotonic()-start)
    (root/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
