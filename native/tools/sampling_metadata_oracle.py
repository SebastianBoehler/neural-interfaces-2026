"""Original sampler oracle from metadata ONLY: no X/y/model/initialization/forward/fit."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import types
import sys

import numpy as np

ROOT=Path('/mnt/lustre/work/gehler/gfh527')
DATA_OWNER=ROOT/'ni26_oscillatory331_v2_20261117/data331.py'
SOURCE_OWNER=ROOT/'ni26_pretrain337_pilot_v3_20261117/common.py'
BASE=ROOT/'ni26_raw312_cpu_20261117_v2'
CONTROL=ROOT/'ni26_combinations338_v2_20261117/output/source6/A/record.json'
EXPECTED='882b5b97ec811d697540f6152be0d02eb49448e9e4c88c3d6c28fcb1433b2f1f'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def function(path,name,owner=None):
    tree=ast.parse(path.read_text())
    if owner:tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def execute(node,env):
    # Only selected original metadata assignments or the pure _batches method.
    if any(isinstance(n,ast.Constant) and n.value in ('X','y') for n in ast.walk(node)):raise ValueError('forbidden X/y access in extracted AST')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),'original_metadata_AST','exec'),env)


def state(rng):
    s=rng.bit_generator.state;a=s['state'];return [a['state']>>64,a['state']&((1<<64)-1),a['inc']>>64,a['inc']&((1<<64)-1),s['has_uint32'],s['uinteger']]


def metadata():
    # Do not execute data331.load/common.source: they load X or selected labels.
    with np.load(BASE/'data/proteus_windows.npz',allow_pickle=False) as z:
        d={k:z[k].copy() for k in ('subject','session')}
    env={'np':np,'d':d};load=function(DATA_OWNER,'load')
    for node in load.body:
        if not isinstance(node,ast.Assign):continue
        if any(isinstance(t,ast.Name) and t.id in ('people','fit') or isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='d' and isinstance(t.slice,ast.Constant) and t.slice.value=='stream' for t in node.targets):execute(node,env)
    env['allfit']=env.pop('fit')
    tree=ast.parse(SOURCE_OWNER.read_text())
    fit_literal=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIT' for t in n.targets))
    execute(fit_literal,env)
    source=function(SOURCE_OWNER,'source')
    fit_node=next(n for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fit' for t in n.targets));execute(fit_node,env)
    fit=env['fit'];env.update(idx=fit,streams=d['stream'][fit],st=d['stream'][fit])
    deep_fit=function(BASE/'experiments/deep.py','_fit','AlignedNet')
    order_node=next(n for n in deep_fit.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='order' for t in n.targets));execute(order_node,env)
    group_fit=function(BASE/'experiments/session_bn.py','_fit','SessionNet')
    env['self']=types.SimpleNamespace(group=32,batch=64,d=d)
    st_node=next(n for n in group_fit.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='st' for t in n.targets));execute(st_node,env)
    sess_node=next(n for n in group_fit.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='_row_sess' for t in n.targets));execute(sess_node,env)
    method=function(BASE/'experiments/session_bn.py','_batches','SessionNet');execute(method,env)
    if len(fit)!=1898 or set(env['people'][fit])!={'P04','P05','P06','P07','P08','P10'} or digest(fit)!=EXPECTED or digest(env['order'])!=EXPECTED:raise ValueError('actual source6 mask/order differs from original proof')
    return d,env,method


def primitive_oracle(path):
    cases=[]
    for seed in (0,1,5,2**32,2**64-1):
        rng=np.random.default_rng(seed);ops=[('STATE',[],state(rng))]
        definitions=[('U64',[])]*3+[('U32',[])]*5+[('U64',[])]*2+[('U32',[])]*3
        definitions += [('PERM',[n]) for n in (0,1,2,31,32,33,190)]
        definitions += [('CHOICE',list(v)) for v in ((0,0,0),(1,31,1),(2,30,1),(3,29,1),(31,1,0),(190,2,0),(190,31,0),(20,20,0),(10001,201,0))]
        definitions += [('BOUND',[n]) for n in (0,1,2**32-2,2**32-1,2**32,2**64-1)]
        for name,args in definitions:
            if name=='U64':value=[int(rng.bit_generator.random_raw())]
            elif name=='U32':value=[int(rng.bit_generator.ctypes.next_uint32(rng.bit_generator.ctypes.state))]
            elif name=='PERM':value=rng.permutation(*args).tolist()
            elif name=='CHOICE':value=rng.choice(args[0],args[1],replace=bool(args[2])).tolist()
            else:value=[int(rng.integers(0,args[0],dtype=np.uint64,endpoint=True))]
            ops.extend([(name,args,value),('STATE',[],state(rng))])
        cases.append((seed,ops))
    with path.open('w') as f:
        f.write(str(len(cases))+'\n')
        for seed,ops in cases:
            f.write(f'{seed} {len(ops)}\n')
            for name,args,value in ops:f.write(' '.join(map(str,[name,*args,len(value),*value]))+'\n')


def main(out):
    if 'torch' in sys.modules:raise ValueError('metadata-only oracle must not import Torch')
    if np.__version__!='2.5.4' or np.version.git_revision!='a3d0ef1df045b858e68115001fdabc930bb3b68c':raise ValueError('installed NumPy source identity differs')
    d,env,method=metadata();receiver=env['self'];rows=np.arange(len(env['order']),dtype=np.int64);rng=np.random.default_rng(0)
    if type(rng.bit_generator).__name__!='PCG64':raise ValueError('default_rng is not original PCG64')
    card=json.loads(CONTROL.read_text());trace=card['trace'];hyper=trace['hyper']
    if any(hyper[k]!=v for k,v in dict(epochs=40,batch=64,group=32,seed=0).items()) or trace['source_order_sha256']!=EXPECTED or card['source_rows_sha256']!=EXPECTED:raise ValueError('original source/seed/hyper authority changed')
    batches=[];states=[state(rng)];epoch_hash=[]
    # Original _train counts nb using a SEPARATE fresh RNG; never consume training RNG.
    count=len(env['_batches'](receiver,rows,np.random.default_rng(0)))
    for epoch in range(40):
        plan=env['_batches'](receiver,rows,rng)
        if len(plan)!=31 or count!=31:raise ValueError('original40x31 plan differs')
        epoch_hash.append(digest(np.concatenate([env['order'][b] for b in plan])))
        if epoch_hash[-1]!=trace['batches'][epoch]:raise ValueError('historical original batch hash mismatch')
        for batch in plan:
            if len(batch)%32 or any(len(np.unique(receiver._row_sess[chunk]))!=1 for chunk in batch.reshape(-1,32)):raise ValueError('original session chunk purity broken')
        batches.extend(plan);states.append(state(rng))
    out.mkdir(exist_ok=False)
    with (out/'source.txt').open('w') as f:
        f.write(f'{len(rows)} 32 64 0 40\n');f.write(' '.join(map(str,receiver._row_sess))+'\n')
        for i in range(40):
            f.write('31\n')
            for batch in batches[i*31:(i+1)*31]:f.write(str(len(batch))+' '+' '.join(map(str,batch))+'\n')
            f.write(' '.join(map(str,states[i+1]))+'\n')
    padded=np.full((1240,64),-1,np.int64);sizes=np.array([len(b) for b in batches],np.int64)
    for i,b in enumerate(batches):padded[i,:len(b)]=b
    primitive_oracle(out/'rng.txt')
    names=list(dict.fromkeys(d['stream'][env['fit']]))
    with (out/'grouping.txt').open('w') as f:
        f.write(f'{len(d["stream"])} {len(rows)}\n');f.write('\n'.join(map(str,d['stream']))+'\n')
        for values in (env['fit'],env['order'],receiver._row_sess):f.write(' '.join(map(str,values))+'\n')
    np.savez(out/'metadata.npz',source_fit=env['fit'],source_order=env['order'],row_session=receiver._row_sess,stream_names=np.asarray(names),batch_indices=padded,batch_sizes=sizes,rng_states=np.asarray(states,np.uint64),group=np.int64(32),batch=np.int64(64),seed=np.uint64(0),epochs=np.int64(40))
    proof=dict(status='ORIGINAL_METADATA_ONLY_40X31_ORACLE',source_rows=1898,source_people=list(env['FIT']),source7_before_filter=sorted(set(env['people'][env['allfit']])),source_fit_sha256=digest(env['fit']),source_order_sha256=digest(env['order']),row_session_sha256=digest(receiver._row_sess),sessions={name:int(np.sum(d['stream'][env['fit']]==name)) for name in names},epoch_global_row_hashes=epoch_hash,batch_sizes=sorted(set(map(int,sizes))),total_batches=1240,original_method_sha256=hashlib.sha256(ast.dump(method).encode()).hexdigest(),owners={str(p):sha(p) for p in (DATA_OWNER,SOURCE_OWNER,BASE/'experiments/session_bn.py',BASE/'experiments/deep.py',CONTROL)},numpy_version=np.__version__,numpy_revision=np.version.git_revision,decoded_dataset_keys=['subject','session'],X_y_access=False,Torch_imported=('torch' in sys.modules),neural_initialization=False,forward=False,fit=False,source_update=False,accuracy=False)
    (out/'proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k not in ('epoch_global_row_hashes','owners')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out)
