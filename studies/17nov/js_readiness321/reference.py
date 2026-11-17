"""Use immutable §304 original training/export paths; never substitute a recipe."""
import ast
import copy
import hashlib
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import torch

SOURCE=Path('/mnt/lustre/work/gehler/gfh527/ni26_s304_coral_prod02_20261116')
MANIFEST='94a5f8a19c3532017f42fc1e85520e9ca0aa511e2ece1bc40df9f406a65b45aa'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def original():
    if sha(SOURCE/'coral304.sha256')!=MANIFEST:
        raise ValueError('original304 source manifest changed')
    subprocess.run(['sha256sum','-c','coral304.sha256'],cwd=SOURCE,check=True,stdout=subprocess.DEVNULL)
    path=SOURCE/'studies/16nov/followups/coral_repair/produce.py'
    spec=importlib.util.spec_from_file_location('original_cc304_producer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.install(module.SB)
    return module


def tracer():
    # Execute exactly the owning exporter helper, without importing its unrelated package builder.
    path=Path.cwd()/'export_external_teacher.py'
    module=ast.parse(path.read_text())
    functions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='device_free_trace']
    if len(functions)!=1:raise ValueError('owning trace helper anchor changed')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['device_free_trace']


def compare(output,reference):
    differences=[];changed=0;count=0
    for order in ('within0','within1','within2','within3','full0','full1','runs0','runs1'):
        if not np.array_equal(output[order+'|index'],reference[order+'|index']):
            raise ValueError('reference canonical/delivery rows differ')
        a,b=output[order+'|proba'],reference[order+'|proba']
        differences.append(float(np.abs(a-b).max()))
        margin=np.sort(b,axis=1)[:,-1]-np.sort(b,axis=1)[:,-2]
        eligible=margin>2e-5
        changed+=int(((a.argmax(1)!=b.argmax(1))&eligible).sum());count+=int(eligible.sum())
    return dict(max_probability_error=max(differences),changed_non_tied=changed,non_tied_rows=count,
                passed=max(differences)<=1e-5 and changed==0)


def export(model,module,path,inputs):
    model.net.cpu().eval();model.dev='cpu';model.set_k(2.)
    stream=module.SB.to_stream(model.net,2.)
    traced=tracer()(stream,torch.from_numpy(inputs[:4]))
    module.SB.reset_stream(traced);traced.save(str(path))
    result={};eager=[]
    with torch.no_grad():
        for prefix in (17,min(61,len(inputs)),len(inputs)):
            x=torch.from_numpy(inputs[:prefix])
            expected=torch.softmax(model.net(x).double(),1).numpy()
            module.SB.reset_stream(traced)
            got=torch.cat([traced(x[i:i+1]) for i in range(len(x))]).double().softmax(1).numpy()
            result[str(prefix)]=float(np.abs(got-expected).max());eager.append(expected)
    module.SB.reset_stream(traced)
    saved=torch.jit.load(str(path));module.SB.reset_stream(saved)
    state=[b for n,b in saved.named_buffers() if n.rsplit('.',1)[-1] in module.SB.STATE]
    if not state or any(float(b.abs().sum())!=0 for b in state):raise ValueError('exported state is not reset')
    left,right=copy.deepcopy(saved),copy.deepcopy(saved)
    module.SB.reset_stream(left);module.SB.reset_stream(right)
    with torch.no_grad():
        x=torch.from_numpy(inputs);cut=min(17,len(x))
        prefix=[left(x[i:i+1]) for i in range(cut)]
        for i in range(min(5,len(x))):right(x[-1-i:len(x)-i])
        combined=torch.cat(prefix+[left(x[i:i+1]) for i in range(cut,len(x))]).double().softmax(1).numpy()
    if float(np.abs(combined-eager[-1]).max())>=1e-4:
        raise ValueError('interleaved session deepcopy shares state')
    if max(result.values())>=1e-4:raise ValueError('CPU stream export differs from its eager source')
    # Existing owning exporter permits1e-3 for CUDA-vs-eager float32 shape differences.
    gpu=torch.jit.load(str(path),map_location='cuda');module.SB.reset_stream(gpu)
    with torch.no_grad():
        x=torch.from_numpy(inputs).cuda()
        got=torch.cat([gpu(x[i:i+1]) for i in range(len(x))]).double().softmax(1).cpu().numpy()
    cuda_error=float(np.abs(got-eager[-1]).max())
    if cuda_error>=1e-3:raise ValueError('CUDA exported stream exceeds owning exporter tolerance')
    return dict(cpu_probability_errors=result,cuda_probability_error=cuda_error,saved_reset=True,deepcopy_interleaved=True,
                stream_buffers=len(state),trace_sha256=sha(path))
