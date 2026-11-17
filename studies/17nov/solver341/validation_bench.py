"""Paired old/new checked API and exact CPU6fullsolve after one validation-only change."""
import json
import os
import resource
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits,threadpool_info
from bindings import load,sha,ROOT


def main():
    root=Path(os.environ['SOLVER_ROOT']);started=time.monotonic();old,x,y,theta,original=load()
    for line in (root/'build/library.sha256').read_text().splitlines():
        pin,path=line.split(maxsplit=1)
        if sha(path)!=pin:raise ValueError('newlibrary/generatedsourcechanged')
    import sys
    sys.path.insert(0,str(ROOT/'studies/16nov/followups/head_kernel'));from kernel import Kernel
    path=root/'build/libheadvalidation.so';new=Kernel(path);checks={}
    for name,xx,yy in (('mini32',x[:32],y[:32]),('full',x,y)):
        a,ga=old.objective(xx,yy,theta,False);b,gb=new.objective(xx,yy,theta,False)
        if max(abs(a-b),float(np.abs(ga-gb).max()))>1e-9:raise ValueError('old/newfixedmathdiffers')
        checks[name]=dict(loss_gap=abs(a-b),gradient_gap=float(np.abs(ga-gb).max()))
    rejected=[]
    for value in (np.nan,np.inf,-np.inf):
        invalid=x[:32].copy();invalid[0,0]=value
        for name,lib in (('old',old),('new',new)):
            try:lib.objective(invalid,y[:32],theta,False)
            except RuntimeError as e:
                if 'nonfinite matrix' not in str(e):raise
                rejected.append(dict(method=name,value=str(value),message=str(e)))
            else:raise ValueError('nonfinite sourceperturbationaccepted')
    times=[]
    for threads in (1,6):
        with threadpool_limits(limits=threads):
            torch.set_num_threads(threads)
            for lib in (old,new):
                for _ in range(2):lib.objective(x,y,theta,False)
            for order in (('old','new'),('new','old'),('old','new')):
                for name in order:
                    tick=time.perf_counter();loss,grad={'old':old,'new':new}[name].objective(x,y,theta,False)
                    times.append(dict(threads=threads,method=name,seconds=time.perf_counter()-tick,objective=loss,max_gradient=float(np.abs(grad).max())))
    fits=[];parameters={}
    with threadpool_limits(limits=6):
        torch.set_num_threads(6)
        for name,lib in (('old',old),('new',new)):
            print('FULLFIT started',name,flush=True);tick=time.perf_counter();fitted,loss,grad,calls=lib.fit(x,y,False);elapsed=time.perf_counter()-tick
            actual,g=old.objective(x,y,fitted,False);independent,ig=old.objective(x,y,fitted,True)
            if max(abs(actual-loss),abs(independent-loss),float(np.abs(g-ig).max()))>1e-9 or max(grad,float(np.abs(g).max()))>1e-5:raise ValueError('fullsolvequality/parity')
            parameters[name]=fitted;fits.append(dict(method=name,seconds=elapsed,objective=loss,max_gradient=grad,calls=calls));print('FULLFIT done',name,elapsed,calls,flush=True)
    difference=float(np.abs(parameters['old']-parameters['new']).max())
    if difference>1e-12 or fits[0]['calls']!=fits[1]['calls'] or abs(fits[0]['objective']-fits[1]['objective'])>1e-12:raise ValueError('validationpatchchangedfitsemantics')
    output=dict(source_sha256=sha(root/'source.sha256'),old_library_sha256=original['library_sha256'],new_library_sha256=sha(path),matrix_sha256=original['matrix_sha256'],
        checks=checks,nonfinite_rejections=rejected,checked_API_timings=times,full_fits=fits,full_parameter_max_abs=difference,
        exact_fitted_parameters=np.array_equal(parameters['old'],parameters['new']),node=os.uname().nodename,affinity=sorted(os.sched_getaffinity(0)),
        threads=6,threadpools=threadpool_info(),seconds=time.monotonic()-started,RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        no_accuracy=True,no_GPU=True,no_provider=True,scope='validationonly/customC++sourceAPI;noactivekerneladoption')
    (root/'validation_result.json').write_text(json.dumps(output));print(json.dumps(fits),flush=True)


if __name__=='__main__':main()
