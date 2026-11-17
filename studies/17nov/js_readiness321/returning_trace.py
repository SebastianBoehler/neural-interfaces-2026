"""Owning reset helper and immutable-state checks without importing training dependencies."""
import ast
import hashlib
from pathlib import Path

SOURCE=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_returning2/source/experiments/session_bn.py')
tree=ast.parse(SOURCE.read_text())
assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STATE' for t in n.targets))
STATE=ast.literal_eval(assignment.value)
function=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reset_stream']
if STATE!=('s1','s2','n') or len(function)!=1:raise ValueError('owning reset source changed')
namespace={'STATE':STATE}
exec(compile(ast.Module(body=function,type_ignores=[]),str(SOURCE),'exec'),namespace)
reset_stream=namespace['reset_stream']


def immutable(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        if k.rsplit('.',1)[-1] not in STATE:
            a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def zero(model):
    if any(float(v.abs().sum()) for k,v in model.named_buffers() if k.rsplit('.',1)[-1] in STATE):raise ValueError('session buffer not zero')
