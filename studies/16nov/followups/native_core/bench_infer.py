"""Matched raw-window Python/native B=1 timing; no accuracy or selection."""
import concurrent.futures
import ctypes
import hashlib
import json
import multiprocessing
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt_zi, sosfiltfilt
from scipy.special import softmax

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'experiments')]
os.environ['NI26_ROOT'] = str(ROOT)
import ts_core2 as T  # noqa: E402
from bridge import Native, pointer  # noqa: E402
from test_preprocess import BANDS  # noqa: E402


def load():
    with np.load(os.environ['S301_BENCH_REFERENCE'], allow_pickle=False) as z:
        return {k: np.ascontiguousarray(z[k]) for k in z.files}


def coeffs():
    band = np.ascontiguousarray([butter(4, list(b), btype='band', fs=120, output='sos') for b in BANDS])
    return (band, np.ascontiguousarray([sosfilt_zi(s) for s in band]),
            np.ascontiguousarray(butter(4, 1, btype='low', fs=120, output='sos')),
            np.ascontiguousarray(sosfilt_zi(butter(4, 1, btype='low', fs=120, output='sos'))))


def execute(method, device, inputs, limit=62):
    started = time.perf_counter()
    native, constants = Native(), coeffs()
    config = T.parse('car_ov17_fs2_t0.1_le_k2')
    prior = inputs['prior']
    count, logsum, slow_sum = 0, np.zeros_like(prior), np.zeros(328, np.float64)
    handle = native.core.nc_create(pointer(prior), device) if method == 'native' else None
    if method == 'native' and not handle:
        raise RuntimeError(native.core.nc_error().decode())
    initialized = time.perf_counter() - started
    output, latency, components = [], [], []
    try:
        for raw in inputs['raw'][:limit]:
            t0 = time.perf_counter()
            if method == 'native':
                cov, slow = native.preprocess(np.ascontiguousarray(raw), constants)
            elif method == 'python_cachedcoeff':
                x = (raw[None] - raw[None].mean(1, keepdims=True)).astype(np.float32)
                cov = []
                for coefficients in constants[0]:
                    band = sosfiltfilt(coefficients, x, axis=-1).astype(np.float32)
                    cov.extend(T.estimate(s, config['est'])[0] for s in (band, band[..., :240], band[..., 240:]))
                cov = np.asarray(cov)
                slow = sosfiltfilt(constants[2], x, axis=-1)[..., 30::60].reshape(328).astype(np.float32).astype(np.float64)
            else:
                x = (raw[None] - raw[None].mean(1, keepdims=True)).astype(np.float32)
                cov = np.asarray([c[0] for c in T.blocks(x, config)])
                slow = T.slow_waveform(x, cutoff=1.0).astype(np.float64)[0]
            t1 = time.perf_counter()
            if method == 'native':
                features = np.empty(44239, np.float64)
                native.check(native.core.nc_step(handle, pointer(cov), pointer(slow), pointer(features)), 'nc_error')
            else:
                count += 1
                for b in range(51):
                    logsum[b] += T.FT.logm(cov[b])
                slow_sum += slow
                ref = T.FT.expm_sym((2 * prior + logsum) / (2 + count))
                tangent = np.concatenate([T.FT.whiten_tangent(cov[b], ref[b]) for b in range(51)])
                features = np.r_[tangent, slow - slow_sum / (2 + count)]
            t2 = time.perf_counter()
            if method == 'native':
                p = native.predict(features[None], inputs['mean'], inputs['scale'], inputs['coef'], inputs['intercept'])[0]
            else:
                logits = ((features - inputs['mean']) / inputs['scale']) @ inputs['coef'].T + inputs['intercept']
                p = softmax(logits)
            t3 = time.perf_counter()
            output.append(p); latency.append(t3 - t0); components.append((t1 - t0, t2 - t1, t3 - t2))
    finally:
        if handle:
            native.core.nc_destroy(handle)
    output = np.asarray(output)
    delta = float(np.max(np.abs(output - inputs['expected_probabilities'][:limit])))
    if delta > 1e-5:
        raise ValueError(f'{method}/{device}: reference probability parity failed: {delta}')
    warm = np.asarray(latency[5:]); comp = np.asarray(components[5:])
    return {'method': method, 'device': device, 'windows': limit, 'initialization_s': initialized,
            'full_session_s': time.perf_counter() - started, 'warm_median_ms': float(np.median(warm) * 1000),
            'warm_p95_ms': float(np.percentile(warm, 95) * 1000),
            'warm_component_mean_ms': {'raw_filter_cov': float(comp[:, 0].mean() * 1000),
                                       'tangent_and_transfer': float(comp[:, 1].mean() * 1000),
                                       'head': float(comp[:, 2].mean() * 1000)},
            'max_probability_diff': delta}


def one_workload(_):
    return execute('native', 0, load(), limit=10)['full_session_s']


def cached(device, inputs):
    with np.load(os.environ['S301_BENCH_CACHE'], allow_pickle=False) as z:
        rows, cov, slow = z['rows'], z['cov'], z['slow']
        metadata = json.loads(str(z['metadata'].item()))
        producer_lock = Path(os.environ['S301_BENCH_PRODUCER']) / 'producer.sha256'
        if (metadata['person'] != 'P01' or metadata['held_accuracy_computed'] is not False or
                metadata['source_sha256'] != hashlib.sha256(producer_lock.read_bytes()).hexdigest()):
            raise ValueError('cached P01 provenance differs')
        index = np.searchsorted(rows, inputs['rows'])
        if not np.array_equal(rows[index], inputs['rows']):
            raise ValueError('cached rows differ from benchmark reference')
        cov, slow = np.ascontiguousarray(cov[index]), np.ascontiguousarray(slow[index])
    start = time.perf_counter(); native = Native()
    handle = native.core.nc_create(pointer(inputs['prior']), device)
    if not handle:
        raise RuntimeError(native.core.nc_error().decode())
    initialized = time.perf_counter() - start
    latency, outputs = [], []
    try:
        for i in range(len(cov)):
            t0 = time.perf_counter()
            feature = np.empty(44239, np.float64)
            native.check(native.core.nc_step(handle, pointer(cov[i]), pointer(slow[i]), pointer(feature)), 'nc_error')
            outputs.append(native.predict(feature[None], inputs['mean'], inputs['scale'], inputs['coef'], inputs['intercept'])[0])
            latency.append(time.perf_counter() - t0)
    finally:
        native.core.nc_destroy(handle)
    diff = float(np.max(np.abs(np.asarray(outputs) - inputs['expected_probabilities'])))
    if diff > 1e-5:
        raise ValueError(f'cached native parity failed: {diff}')
    return {'method': 'native_cached', 'device': device, 'windows': len(cov), 'initialization_s': initialized,
            'full_session_s': time.perf_counter() - start, 'warm_median_ms': float(np.median(latency[5:]) * 1000),
            'warm_p95_ms': float(np.percentile(latency[5:], 95) * 1000), 'max_probability_diff': diff,
            'person_preparation_s': metadata['seconds'],
            'preparation_scope': 'all person windows: covariances, log-prior and whole-session training features; includes more than inference cache'}


def main():
    target = Path(os.environ['S301_BENCH_OUTPUT'])
    if target.exists():
        raise FileExistsError(target)
    ref_path = Path(os.environ['S301_BENCH_REFERENCE'])
    reference_lock = json.loads((ref_path.parent / 'manifest.json').read_text())
    if (reference_lock['held_accuracy_computed'] is not False or
            hashlib.sha256(ref_path.read_bytes()).hexdigest() != reference_lock['files'][ref_path.name]):
        raise ValueError('benchmark reference checksum mismatch')
    smoke = json.loads(Path(os.environ['S301_BENCH_SMOKE']).read_text())
    if (smoke['held_accuracy_computed'] is not False or
            smoke['benchmark_sha256'] != hashlib.sha256((ROOT / 'benchmark.sha256').read_bytes()).hexdigest() or
            smoke['reference_sha256'] != hashlib.sha256(ref_path.read_bytes()).hexdigest() or
            smoke['native_core_sha256'] != hashlib.sha256(Path(os.environ['NATIVE_CORE_LIB']).read_bytes()).hexdigest()):
        raise ValueError('benchmark smoke gate differs')
    inputs = load()
    timing = []
    for pair in ((('python', 0), ('native', 0)), (('native', 0), ('python', 0)), (('python', 0), ('native', 0))):
        for method, device in pair:
            timing.append(execute(method, device, inputs))
    for _ in range(3):
        timing.append(execute('native', 1, inputs))
    for device in (0, 1):
        for _ in range(3):
            timing.append(cached(device, inputs))
    throughput = []
    for workers in (1, 2, 4):
        start = time.perf_counter(); usage0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            individual = list(pool.map(one_workload, range(4)))
        elapsed = time.perf_counter() - start
        usage1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        throughput.append({'workers': workers, 'jobs': 4, 'windows_per_job': 10, 'elapsed_s': elapsed,
                           'completed_jobs_per_hour': 4 * 3600 / elapsed,
                           'child_cpu_seconds': (usage1.ru_utime + usage1.ru_stime) - (usage0.ru_utime + usage0.ru_stime),
                           'child_peak_rss_kib': usage1.ru_maxrss, 'individual_session_s': individual})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f:
        json.dump({'timings': timing, 'throughput': throughput,
                   'reference_sha256': hashlib.sha256(Path(os.environ['S301_BENCH_REFERENCE']).read_bytes()).hexdigest(),
                   'native_core_sha256': hashlib.sha256(Path(os.environ['NATIVE_CORE_LIB']).read_bytes()).hexdigest(),
                   'native_train_sha256': hashlib.sha256(Path(os.environ['NATIVE_TRAIN_LIB']).read_bytes()).hexdigest(),
                   'benchmark_source_sha256': hashlib.sha256((ROOT / 'benchmark.sha256').read_bytes()).hexdigest(),
                   'resource_scope': 'one 4-CPU Slurm allocation; 1/2/4 workers use same total CPUs',
                   'memory_note': 'Slurm cgroup MaxRSS includes the whole allocation; child_peak_rss is per largest child, not their sum',
                   'held_accuracy_computed': False}, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'matched raw-to-posterior timings saved: {target}; no score', flush=True)


if __name__ == '__main__':
    main()
