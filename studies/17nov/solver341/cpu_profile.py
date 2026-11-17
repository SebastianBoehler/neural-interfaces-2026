"""Same-binary/thread paired full solves plus actual C++ operator profile, no accuracy."""
import json
import os
import platform
import resource
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_info,threadpool_limits
from bindings import load,sha,MATRIX_SHA,LIBRARY_SHA


def operators(profile,memory):
    records=[]
    for e in profile.key_averages():
        records.append(dict(name=e.key,count=e.count,self_cpu_ms=e.self_cpu_time_total/1000,total_cpu_ms=e.cpu_time_total/1000,
            self_memory_bytes=e.self_cpu_memory_usage if memory else None))
    records.sort(key=lambda x:x['self_cpu_ms'],reverse=True)
    if not any(e['name'] in ('aten::mm','aten::matmul','aten::addmm') for e in records):raise ValueError('profiler did notobserveactualC++matmul')
    return records


def check(lib,x,y,theta):
    a,ga=lib.objective(x,y,theta,False);b,gb=lib.objective(x,y,theta,True)
    if not np.isfinite([a,b]).all() or not np.isfinite(ga).all() or max(abs(a-b),float(np.abs(ga-gb).max()))>1e-9:raise ValueError('independent analytical/autograd parity')
    return dict(objective=a,max_gradient=float(np.abs(ga).max()),loss_gap=abs(a-b),gradient_gap=float(np.abs(ga-gb).max()))


def main():
    root=Path(os.environ['SOLVER_ROOT']);started=time.monotonic();lib,x,y,theta,original=load();fixed={};profiles={};timings=[];fits=[]
    if len(os.sched_getaffinity(0))<6 or torch.cuda.is_available():raise ValueError('explicit6CPU/noCUDAallocation required')
    for threads in (1,6):
        with threadpool_limits(limits=threads):
            torch.set_num_threads(threads)
            fixed[str(threads)]={'mini':check(lib,x[:32],y[:32],theta),'full':check(lib,x,y,theta)}
            for _ in range(2):lib.objective(x,y,theta,False)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,profile_memory=True) as p:lib.objective(x,y,theta,False)
            profiles[str(threads)]=operators(p,True)
    for order in ((1,6),(6,1),(1,6)):
        for threads in order:
            with threadpool_limits(limits=threads):
                torch.set_num_threads(threads);tick=time.perf_counter();loss,grad=lib.objective(x,y,theta,False)
                timings.append(dict(threads=threads,seconds=time.perf_counter()-tick,objective=loss,gradient=float(np.abs(grad).max())))
    target=fixed['1']['full']['objective']+1e-5;parameters={}
    (root/'preflight.json').write_text(json.dumps(dict(passed=True,fixed=fixed,checked_API_profile=profiles,checked_API_times=timings)))
    for threads in (1,6):
        with threadpool_limits(limits=threads):
            torch.set_num_threads(threads);print('FULLFIT started threads',threads,flush=True);tick=time.perf_counter()
            fitted,loss,gradient,calls=lib.fit(x,y,False);elapsed=time.perf_counter()-tick;identity=check(lib,x,y,fitted)
            if loss>target or gradient>1e-5 or identity['max_gradient']>1e-5 or abs(loss-identity['objective'])>1e-9:raise ValueError('matchedfullsolvequality')
            parameters[threads]=fitted
            fits.append(dict(threads=threads,seconds=elapsed,objective=loss,max_gradient=gradient,evaluations=calls,independent=identity,threadpools=threadpool_info()))
            (root/'fit_progress.json').write_text(json.dumps(fits));print('FULLFIT done',threads,elapsed,calls,flush=True)
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);print('PROFILEDFULLFIT started',flush=True);tick=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False,profile_memory=False) as p:
            fitted,loss,gradient,calls=lib.fit(x,y,False)
        profiled_seconds=time.perf_counter()-tick;fit_ops=operators(p,False)
        if calls!=fits[0]['evaluations'] or np.max(np.abs(fitted-parameters[1]))>1e-12 or abs(loss-fits[0]['objective'])>1e-12:raise ValueError('profilerchangedfitsemantics')
    with open('/proc/self/maps') as f:mapped=sorted({s.split()[-1] for s in f if '/' in s and any(k in s.lower() for k in ('torch','mkl','blas','gomp','iomp'))})
    output=dict(source_sha256=sha(root/'source.sha256'),matrix_sha256=MATRIX_SHA,library_sha256=LIBRARY_SHA,source_rows_sha256=original['rows_sha256'],
        original_options=dict(C=.3,max_iter=2000,max_eval=2500,history=10,tolerance_grad=1e-8,tolerance_change=1e-12,line_search='strong_wolfe'),
        fixed_checks=fixed,checked_API_times=timings,checked_API_profiles=profiles,full_solves=fits,profiled_full_seconds=profiled_seconds,
        profiled_full_operators=fit_ops,profiled_full_semantics_exact=True,source_shape=list(x.shape),source_matrix_bytes=x.nbytes,
        affinity=sorted(os.sched_getaffinity(0)),platform=platform.platform(),node=os.uname().nodename,torch=torch.__version__,torch_config=torch.__config__.show(),
        mapped_libraries=mapped,process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.monotonic()-started,
        no_held_accuracy=True,no_Weco_or_provider=True,scope='sameunscoredsource/customC++binary/CPUhost/threadcount;NOTlanguage/device/accuracyclaim')
    (root/'result.json').write_text(json.dumps(output));print('COMPLETE',json.dumps(fits),flush=True)


if __name__=='__main__':main()
