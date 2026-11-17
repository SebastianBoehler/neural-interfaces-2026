"""Complete label-free held streams with the exact frozen native causal producer."""
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from bindings import sha
from held import BASE as FIXTURES, compare
from reference import FullReference
from second_fold import load as load_p10, PRODUCER, REFERENCES
from sklearn_reference import fit as sklearn_fit
from verify import objective, parity

BASE = PRODUCER.parent
V4 = BASE/'ni26_solver343_layout_v4_20261117'
CORE = BASE/'ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so'
LIBRARY_SHA = '7a2715ad8b912a8c464d304ca1a671bed9dd889deaf5b8bf99907fe6efa939bd'


def heads():
    if sha(V4/'build/libspan.so') != LIBRARY_SHA:
        raise ValueError('qualified library changed')
    proof = json.loads((V4/'reference_benchmark_result.json').read_text())
    if not proof['passed'] or proof['library_sha256'] != LIBRARY_SHA:
        raise ValueError('complete two-fold/prefix benchmark must pass')
    baseline = BASE/'ni26_solver343_baseline_v1_20261117/cold_sklearn_1.npz'
    p01 = (np.load(baseline)['theta'].copy(), np.load(V4/'reference_full_fit.npz')['theta'].copy())
    x, y, provenance = load_p10()
    sklearn, options = sklearn_fit(x, y)
    reference = np.column_stack((sklearn.coef_, sklearn.intercept_))
    candidate, stats, _ = FullReference(V4/'build/libspan.so').fit(x, y)
    loss0, _, p0 = objective(x, y, reference)
    loss, grad, p = objective(x, y, candidate)
    source = parity(p, p0)
    prefix = compare('P10', candidate, reference)
    if loss > loss0+1e-5 or grad > 1e-5 or source['max_probability_error'] > 1e-3 or source['non_tied_changes'] or not all(r['passed'] for r in prefix.values()):
        raise ValueError('P10 source export did not qualify')
    np.savez('source_heads.npz', P01_reference=p01[0], P01_candidate=p01[1],
             P10_reference=reference, P10_candidate=candidate)
    return {'P01':p01,'P10':(reference,candidate)}, dict(P10_source=provenance,
        P10_reference_optimizer=options,P10_candidate_optimizer=stats,P10_source_parity=source,
        P10_objective=loss,P10_max_gradient=grad,source_heads_sha256=sha('source_heads.npz'),
        P01_reference_archive_sha256=sha(baseline),P01_candidate_archive_sha256=sha(V4/'reference_full_fit.npz'))


def core():
    lib = ctypes.CDLL(str(CORE))
    lib.nc_create.argtypes, lib.nc_create.restype = [ctypes.c_void_p,ctypes.c_int],ctypes.c_void_p
    lib.nc_step.argtypes = [ctypes.c_void_p]*4
    lib.nc_destroy.argtypes = [ctypes.c_void_p]
    lib.nc_error.restype = ctypes.c_char_p
    return lib


def features(lib, prior, cov, slow, indices):
    handle = lib.nc_create(prior.ctypes.data,0)
    if not handle: raise RuntimeError(lib.nc_error().decode())
    try:
        for index in indices:
            out = np.empty(44239,np.float64)
            if lib.nc_step(handle,cov[index].ctypes.data,slow[index].ctypes.data,out.ctypes.data):
                raise RuntimeError(lib.nc_error().decode())
            yield out
    finally:
        lib.nc_destroy(handle)


def main():
    tick = time.monotonic()
    if torch.cuda.is_available() or len(os.sched_getaffinity(0)) < 2:
        raise ValueError('explicit bound CPU2 route required')
    torch.set_num_threads(2)
    lib = core()
    with threadpool_limits(limits=2):
        models, source = heads()
        with np.load(PRODUCER/'data/proteus_windows.npz',allow_pickle=False) as z:
            subject, session = z['subject'],z['session']
        streams = np.char.add(np.char.add(subject,'|'),session)
        permfile = PRODUCER/'runs/shuffle_perms.npz'
        with np.load(permfile,allow_pickle=False) as z:
            permutations = {o:z[f'{o[:-1]}|{o[-1]}|all_loso'].copy() for o in ('within0','full0')}
        manifest = json.loads((FIXTURES/'manifest.json').read_text())
        if sha(permfile) != manifest['permutation_sha256']: raise ValueError('frozen permutations differ')
        results, canaries = {},{}
        for phase in ('canary','complete'):
            for held in ('P01','P10'):
                path = PRODUCER/f'runs/s301_native_inputs/{held}.npz'
                with np.load(path,allow_pickle=False) as z:
                    rows,cov,slow = z['rows'].copy(),z['cov'].copy(),z['slow'].copy()
                    metadata = json.loads(str(z['metadata'].item()))
                if metadata['person'] != held or metadata['native_core_sha256'] != sha(CORE) or metadata['held_accuracy_computed'] or metadata['source_sha256'] != sha(PRODUCER/'producer.sha256') or metadata['rows_sha256'] != hashlib.sha256(rows.tobytes()).hexdigest() or not np.all(subject[rows] == f'Dreyer2026Proteus/{held}'):
                    raise ValueError('frozen causal-input provenance differs')
                reference,candidate = models[held]
                with np.load(REFERENCES/f'{held}.npz',allow_pickle=False) as z:
                    mean,scale,fit_rows = z['mean'].copy(),z['scale'].copy(),z['fit_rows'].copy()
                if np.isin(rows,fit_rows).any(): raise ValueError('source/query overlap')
                for order,permutation in permutations.items():
                    if not np.array_equal(np.sort(permutation),np.arange(len(subject))): raise ValueError('invalid delivery permutation')
                    fixture = FIXTURES/f'{held}_{order}.npz'
                    if sha(fixture) != manifest['files'][fixture.name]: raise ValueError('frozen fixture differs')
                    with np.load(fixture,allow_pickle=False) as z:
                        prior,canary_rows,expected = z['prior'].copy(),z['rows'].copy(),z['expected_features'].copy()
                        if not np.array_equal(z['mean'],mean) or not np.array_equal(z['scale'],scale) or not np.array_equal(z['fit_rows'],fit_rows): raise ValueError('actual scaler/fit rows differ')
                    indices = np.searchsorted(rows,canary_rows)
                    key = f'{held}/{order}'
                    if phase == 'canary':
                        observed = np.array(list(features(lib,prior,cov,slow,indices)))
                        absolute = float(abs(observed-expected).max()); relative = float(np.linalg.norm(observed-expected)/np.linalg.norm(expected))
                        if not np.isfinite([absolute,relative]).all() or absolute > 1e-4 or relative > 1e-6: raise ValueError('original Python feature canary failed')
                        canaries[key] = dict(windows=len(indices),feature_max_abs=absolute,feature_relative_L2=relative,fixture_sha256=sha(fixture))
                        continue
                    if len(canaries) != 4: raise ValueError('all four canaries must pass before full generation')
                    delivered = permutation[subject[permutation] == f'Dreyer2026Proteus/{held}']
                    if len(delivered) != len(rows) or not np.array_equal(np.sort(delivered),rows): raise ValueError('complete held coverage differs')
                    outputs, references, delivered_rows = [],[],[]
                    for stream in dict.fromkeys(streams[delivered]):
                        selected = delivered[streams[delivered] == stream]
                        for row,feature in zip(selected,features(lib,prior,cov,slow,np.searchsorted(rows,selected))):
                            value = (feature-mean)/scale
                            references.append(softmax(value@reference[:,:-1].T+reference[:,-1]))
                            outputs.append(softmax(value@candidate[:,:-1].T+candidate[:,-1]));delivered_rows.append(int(row))
                    row = parity(np.array(outputs),np.array(references))
                    row.update(windows=len(outputs),sessions=len(set(streams[delivered])),pinned_order=order,
                               rows_sha256=hashlib.sha256(np.array(delivered_rows,np.int64).tobytes()).hexdigest(),
                               input_sha256=sha(path),source_reference_sha256=sha(REFERENCES/f'{held}.npz'))
                    row['passed'] = row['max_probability_error'] <= 1e-3 and row['non_tied_changes'] == 0
                    results[key] = row
                    Path('full_held_progress.json').write_text(json.dumps(dict(canaries=canaries,results=results),indent=2))
                    print(key,json.dumps(row),flush=True)
    passed = len(results)==4 and all(r['passed'] for r in results.values()) and sum(r['windows'] for r in results.values())==1534
    report = dict(passed=passed,results=results,canaries=canaries,source=source,device=0,
                  no_held_labels_or_accuracy=True,source_manifest_sha256=sha('source.sha256'),
                  core_sha256=sha(CORE),library_sha256=LIBRARY_SHA,permutation_sha256=sha(permfile),
                  node=os.uname().nodename,affinity=sorted(os.sched_getaffinity(0)),seconds=time.monotonic()-tick)
    Path('full_held_result.json').write_text(json.dumps(report,indent=2))
    if not passed: raise ValueError('full held-stream qualification failed')


if __name__ == '__main__': main()
