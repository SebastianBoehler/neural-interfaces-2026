"""Cyclic nested SOURCE8 folds; execute EXACT pinned339 scientific producer body."""
import importlib.util
import inspect
import os
import sys
from pathlib import Path
import numpy as np
OLD=Path('/mnt/lustre/work/gehler/gfh527/ni26_calibration339_parallel_v1_20261117')
sys.path.insert(0,str(OLD))
import shared339 as science
if science.sha(OLD/'source.sha256')!='85d906c98def9fee7dcd5ab34363c51f4411be7b63d25e88b3ff97fa69bf5916':raise ValueError('prototypeproducerchanged')
for line in (OLD/'source.sha256').read_text().splitlines():
    h,n=line.split(maxsplit=1)
    if science.sha(OLD/n)!=h:raise ValueError('prototypecodechanged')
import produce339 as original


def inputs_for(outer,cal):
    if (outer,cal)==('P01','P02'):return science.inputs()
    module=science.inputs339;text=inspect.getsource(module.inputs)
    old="{f'P{i:02d}' for i in range(3,11)}"
    if text.count(old)!=1:raise ValueError('sourcesplitguardanchorchanged')
    text=text.replace(old,'set(src.PEOPLE)-{OUTER,CAL}')
    env=dict(vars(module),OUTER=outer,CAL=cal);exec(compile(text,'339_cyclic_source8_split','exec'),env)
    return env['inputs']()


def main():
    outer=sys.argv[1];seed=int(sys.argv[2]);people=science.src.PEOPLE
    if outer not in people or seed not in (0,1):raise ValueError('unregisteredsplit')
    cal=people[(people.index(outer)+1)%len(people)];root=Path(os.environ['FULL_ROOT']);folder=root/outer;folder.mkdir(exist_ok=True)
    manifest=root/'source.sha256'
    if (folder/'source.sha256').exists():
        if science.sha(folder/'source.sha256')!=science.sha(manifest):raise ValueError('foldsourcechanged')
    else:(folder/'source.sha256').write_bytes(manifest.read_bytes())
    original.OUTER=outer;original.CAL=cal;original.inputs=lambda:inputs_for(outer,cal)
    os.environ['CAL_ROOT']=str(folder);sys.argv=[sys.argv[0],str(seed)]
    # Scheduling benchmark only: a cold epoch contains one-time setup; measure a fresh warm fit.
    text=inspect.getsource(original.main)
    anchor="    projection=1.5*seconds*40+120"
    if text.count(anchor)!=1:raise ValueError('costanchorchanged')
    benchmark="""    cold_seconds=seconds
    del probe,x,y;gc.collect();torch.cuda.empty_cache()
    m.SEED=seed;x,y,scale,mu,sd,cw,groups=m.setup(d,rows,'cuda',epochs=1)
    probe=m.Factor('population',False,'cuda');tick=time.monotonic()
    warm_smoke=m.train(probe,x,y,mu,sd,cw,groups);torch.cuda.synchronize();seconds=time.monotonic()-tick
    if warm_smoke['max_gradient']<=0:raise ValueError('warmprobeinactive')
    projection=max(cold_seconds+1.5*seconds*40+120,2*19.4*len(rows)/2276+60)"""
    text=text.replace(anchor,benchmark).replace('cost=dict(seed=seed,smoke=smoke,seconds=seconds,',
        'cost=dict(seed=seed,smoke=smoke,warm_smoke=warm_smoke,cold_seconds=cold_seconds,seconds=seconds,')
    env=dict(vars(original));exec(compile(text,'339_full_warm_cost_orchestration','exec'),env)
    env['main']()


if __name__=='__main__':main()
