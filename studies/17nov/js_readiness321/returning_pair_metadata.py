"""Independent owning protocol split/cell metadata, without opening any query label array."""
import ast
from pathlib import Path

import numpy as np

from returning_assets import ROOT,sha


def held_metadata(raw):
    path=ROOT/'neural-interfaces-2026/experiments/protocol.py';tree=ast.parse(path.read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('folds','_folds')]
    load=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load')
    stream=[n for n in load.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Constant) and t.slice.value=='stream' for t in n.targets)]
    score=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score')
    cells=[n for n in score.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cells' for t in n.targets)]
    if len(funcs)!=2 or len(stream)!=1 or len(cells)!=1:raise ValueError('actual protocol metadata anchors differ')
    d={k:np.asarray(raw[k]).copy() for k in ('subject','session','interface','split')}
    ns=dict(np=np,d=d);exec(compile(ast.Module(body=funcs+stream,type_ignores=[]),str(path),'exec'),ns)
    result={}
    for fold,train,held in ns['folds'](d,'seen_session'):
        ns['mask']=held;exec(compile(ast.Module(body=cells,type_ignores=[]),str(path),'exec'),ns)
        keys=sorted({(str(s).rsplit('|',1)[1],str(i)) for s,i in ns['cells']})
        result[fold]=dict(rows=np.flatnonzero(held),train=np.flatnonzero(train),cell_keys=[list(k) for k in keys],protocol_sha256=sha(path))
    if set(result)!={'P04_s23','P10_s23'}:raise ValueError('actual owning returning splits differ')
    return result
