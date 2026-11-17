"""Fixed-parameter same-work skinny GEMM layout comparison before any new fits."""
import ctypes
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from bindings import load, sha
from reference import FullReference

OLD = Path('/mnt/lustre/work/gehler/gfh527/ni26_solver343_fused_v3_20261117/build/libspan.so')


def main():
    old, x, y, theta, _ = load()
    torch.set_num_threads(6)
    parameter = np.ascontiguousarray(theta.T).ravel()
    libraries = {name: FullReference(path) for name, path in [('old', OLD), ('classmajor', Path('build/libspan.so').resolve())]}
    handles, cold, values, calls = {}, {}, {}, []
    with threadpool_limits(limits=6):
        for name, model in libraries.items():
            handle, stats = ctypes.c_void_p(), np.empty(7, np.float64)
            model.check(model.lib.s343f_create(x.ctypes.data, y.ctypes.data, *x.shape, .3, ctypes.byref(handle), stats.ctypes.data))
            handles[name] = handle
            tick = time.perf_counter()
            loss, gradient = np.empty(1, np.float64), np.empty_like(parameter)
            model.check(model.lib.s343r_objective(handle, parameter.ctypes.data, loss.ctypes.data, gradient.ctypes.data))
            cold[name] = time.perf_counter()-tick
            values[name] = (float(loss[0]), gradient.copy())
        loss, gradient = old.objective(x, y, theta, False)
        guards = {name: dict(objective_error=abs(v[0]-loss), gradient_error=float(abs(v[1].reshape(-1,3).T-gradient).max())) for name,v in values.items()}
        if any(max(row.values()) > 1e-9 for row in guards.values()):
            raise ValueError('original-space objective/gradient guard failed')
        for order in [('old','classmajor'),('classmajor','old')]*3:
            for name in order:
                model = libraries[name]
                loss, gradient = np.empty(1, np.float64), np.empty_like(parameter)
                tick, cpu = time.perf_counter(), time.process_time()
                model.check(model.lib.s343r_objective(handles[name], parameter.ctypes.data, loss.ctypes.data, gradient.ctypes.data))
                calls.append(dict(method=name, wall_s=time.perf_counter()-tick, CPU_core_s=time.process_time()-cpu))
        for name, model in libraries.items(): model.lib.s343r_destroy(handles[name])
    report = dict(passed=True, guards=guards, cold=cold, paired_calls=calls,
                  libraries={'old':sha(OLD),'classmajor':sha('build/libspan.so')},
                  node=os.uname().nodename, affinity=sorted(os.sched_getaffinity(0)), no_held_accuracy=True)
    Path('layout_result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
