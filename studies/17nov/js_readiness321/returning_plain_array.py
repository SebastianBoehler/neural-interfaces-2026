"""Original parameter exporters and frozen Shallow/FBC runtime against NEW genuine PyTorch fits."""
import ast
import copy
import json
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from returning_plain_collect import ROOT,checked,asset
from returning_bank import input_class
from returning_train import sha


def owners(source):
    functions={};namespace={'np':np}
    for file,func in (('train_ensemble.py','net_params'),('add_fbcnet.py','fbc_params')):
        p=source/'scripts'/file;nodes=[]
        for n in ast.parse(p.read_text()).body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NET_KEYS' for t in n.targets):nodes.append(n)
            if isinstance(n,ast.FunctionDef) and n.name==func:nodes.append(n)
        if not any(isinstance(n,ast.FunctionDef) for n in nodes):raise ValueError('original array exporter missing')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace);functions[func]=namespace[func]
    archive=ROOT/'ni26_js321_20261117_active5/base_v66.zip'
    if sha(archive)!='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed':raise ValueError('frozenZIP changed')
    with zipfile.ZipFile(archive) as z:text=z.read('submission.py').decode()
    names={'Net','Shallow','FBC','fft_bank'};nodes=[n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes}!=names:raise ValueError('original runtime anchors differ')
    namespace=dict(np=np,torch=torch,F=F,INPUT_KEYS={'align','k','G','scale','shrink'})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozenZIP/submission.py','exec'),namespace)
    return functions,namespace


def main():
    torch.set_num_threads(1);start=time.monotonic();source=ROOT/'ni26_js321_20261117_returning2/source'
    sys.path[:0]=[str(source/'experiments'),str(ROOT/'neural-interfaces-2026/2026-competition/tracks/bci_decoding')]
    import deep
    exporters,namespace=owners(source);Input=input_class();records=[]
    data=source/'data/proteus_windows.npz'
    with np.load(data,allow_pickle=True) as z:
        rows=np.flatnonzero((z['subject']=='Dreyer2026Proteus/P04')&(z['session'].astype(str)=='2'))[:31];X=z['X'][rows].copy()
    for cfg in ('shallow_ea','fbcnet_ea_tf'):
        proof=checked(cfg,'P04_s23',0);path=asset(cfg,'P04_s23',0);c=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False);r=c['recipe']
        net=deep.build(r['arch'],41,480,True).eval();net.load_state_dict(c['state'],strict=True)
        m=types.SimpleNamespace(net=net,align=r['align'],k=r['k'],G=c['G'],scale=c['scale'])
        p=exporters['net_params' if cfg.startswith('shallow') else 'fbc_params'](m)
        original=namespace['Shallow' if cfg.startswith('shallow') else 'FBC'](p,'cpu',path);stream=Input(p)
        h=torch.from_numpy(np.stack([stream(x) for x in X]));before={k:v.clone() for k,v in net.state_dict().items()}
        with torch.no_grad():
            ref=torch.cat([net(h[i:i+1]) for i in range(31)]).double().softmax(1).numpy()
            got=torch.cat([original.proba(h[i:i+1].to(dtype=original.dtype),{}) for i in range(31)]).numpy()
            repeat=torch.cat([original.proba(h[i:i+1].to(dtype=original.dtype),{}) for i in range(31)]).numpy()
        error=float(np.abs(ref-got).max());changed=(ref.argmax(1)!=got.argmax(1))&(np.sort(ref,axis=1)[:,-1]-np.sort(ref,axis=1)[:,-2]>2e-4)
        if error>1e-4 or changed.any() or not np.array_equal(got,repeat):raise ValueError(f'original runtime vs genuine PyTorch failed {cfg} {error}')
        if not all(torch.equal(v,net.state_dict()[k]) for k,v in before.items()):raise ValueError('PyTorch state mutated')
        np.savez(cfg+'_arrays.npz',**p);records.append(dict(family=cfg,original_runtime_class=type(original).__name__,maxabs=error,nontied_changes=int(changed.sum()),samepath_bitwise=True,asset=proof,array_sha256=sha(cfg+'_arrays.npz')))
    Path('array_guard.json').write_text(json.dumps(dict(status='original_plain_runtime_source_array_export_passed_NOT_accuracy',cases=records,
        source_sha256=sha('source.sha256'),seconds=time.monotonic()-start,query_labels_read=False,accuracy_computed=False),indent=2))


if __name__=='__main__':main()
