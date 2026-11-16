"""Fit one source-only nested gate and joint heads; stream held EEG without scoring."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate
import head
import inputs
from source_labels import read_selected, read_training_labels
from style import sha

SRC = inputs.PROD
sys.path.insert(0, str(SRC / 'studies/16nov/followups/native_core'))
from cpu_bridge import CpuNative

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
ARCHIVES = {o: ('coremember.npz' if o in ('within0', 'within1', 'full0', 'runs0') else
                 'coremember_within2_within3_full1_runs1.npz') for o in ORDERS}


def check_lock(root):
    path = root / 'runs/input306/inputs.json'
    expected = path.with_suffix('.sha256').read_text().split()[0]
    if sha(path) != expected: raise ValueError('input lock digest mismatch')
    lock = json.loads(path.read_text())
    if lock['held_labels_opened'] is not False or set(lock['files']) != set(map(str, inputs.paths())):
        raise ValueError('input lock incomplete')
    for name, digest in lock['files'].items():
        if sha(name) != digest: raise ValueError(f'input differs: {name}')
    inputs.check_sources()
    return expected, lock['files']


def gate_state(model):
    (scaler, classifier), cal = model
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                weight=classifier.coef_.reshape(-1).tolist(), intercept=float(classifier.intercept_[0]),
                calibration_weight=float(cal.coef_[0, 0]), calibration_bias=float(cal.intercept_[0]))


def main(index, mode):
    root = Path(os.environ['EXPERT_ROOT']).resolve(strict=True)
    if not 0 <= index < 10 or mode not in ('smoke', 'full'): raise ValueError('invalid fold or mode')
    affinity = sorted(os.sched_getaffinity(0))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('SLURM_CPUS_PER_TASK') != '6' or len(affinity) != 6:
        raise ValueError(f'CPU-only six-logical-CPU binding required, got {affinity}')
    torch.set_num_threads(6)
    held = f'P{index + 1:02d}'
    output = root / f'runs/expert306_{mode}/{held}.npz'
    if output.exists(): raise FileExistsError(output)
    locksha, hashes = check_lock(root)
    data = SRC / 'data/proteus_windows.npz'
    with np.load(data, allow_pickle=False) as z:
        subject, session = z['subject'].copy(), z['session'].copy()
    held_id = f'Dreyer2026Proteus/{held}'
    test = subject == held_id; train = ~test
    if len(set(subject[train])) != 9 or held_id in set(subject[train]): raise ValueError('wrong fold')
    fit_rows = np.flatnonzero(train); y = read_training_labels(data, fit_rows, len(subject))
    source_interface = read_selected(data, 'interface', fit_rows, len(subject))
    domains = (source_interface == 'Graz').astype(np.int64)
    with np.load(inputs.STYLE, allow_pickle=False) as z:
        if not np.array_equal(z['index'], np.arange(len(subject))): raise ValueError('style rows mismatch')
        style = z['values'].copy(); style_meta = json.loads(str(z['metadata'].item()))
    if style.shape != (len(subject), 246) or style_meta['labels_opened'] is not False:
        raise ValueError('invalid gate feature cache')
    started = time.monotonic()
    oof, gate_model, trace = gate.nested(style[fit_rows], subject[fit_rows], y, domains, held_id)
    test_gate = gate.predict(gate_model, style[test])
    if np.std(oof) < 1e-8 or np.std(test_gate) < 1e-8: raise ValueError('inactive gate')
    gs = gate_state(gate_model)
    gate_logit = ((style[test] - gs['mean']) / gs['scale']) @ np.asarray(gs['weight']) + gs['intercept']
    frozen_gate = expit(gs['calibration_weight'] * gate_logit + gs['calibration_bias'])
    if not np.allclose(test_gate, frozen_gate, atol=1e-12, rtol=0):
        raise ValueError('exported gate does not reproduce the deployed gate')
    with (inputs.REF / 'manifest.json').open() as f: refs = json.load(f)
    refpath = inputs.REF / f'{held}.npz'
    if sha(refpath) != refs['files'][refpath.name]: raise ValueError('reference head mismatch')
    with np.load(refpath, allow_pickle=False) as z:
        fixed = {k: np.ascontiguousarray(z[k]) for k in z.files}
    if not np.array_equal(fixed['fit_rows'], fit_rows): raise ValueError('reference rows differ')
    native = CpuNative(); prior_parts, counts, features, indexes, saved = [], [], [], [], None
    for i in range(10):
        person = f'P{i + 1:02d}'; p = SRC / f'runs/s301_native_inputs/{person}.npz'
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item())); rows = z['rows'].copy()
            if (meta['person'] != person or meta['source_sha256'] != inputs.EXPECTED['producer.sha256'] or
                    meta['parity_report_sha256'] != inputs.EXPECTED['report.json'] or
                    meta['native_core_sha256'] != inputs.EXPECTED['libnative_core.so'] or
                    meta['native_train_sha256'] != inputs.EXPECTED['libtrainref.so'] or
                    meta['held_accuracy_computed'] is not False or len(rows) == 0 or
                    not np.all(subject[rows] == f'Dreyer2026Proteus/{person}')):
                raise ValueError(f'bad source shard {person}')
            prior_parts.append(z['prior'].copy()); counts.append(len(rows))
            if i == index: saved = {'rows': rows, 'cov': z['cov'].copy(), 'slow': z['slow'].copy()}
            else: indexes.append(rows); features.append(z['features'].copy())
    train_index = np.concatenate(indexes); pos = np.argsort(train_index)
    if saved is None or not np.array_equal(train_index[pos], fit_rows) or not np.array_equal(saved['rows'], np.flatnonzero(test)):
        raise ValueError('cached fold rows differ')
    x = np.ascontiguousarray(np.concatenate(features)[pos]); del features
    x = native.standardize(x, fixed['mean'], fixed['scale'])
    weights = gate.strata_weights(subject[fit_rows], y, domains)
    signed = 2 * oof - 1
    shared, shared_stats = head.fit(x, y, signed, weights, 1., shared=True)
    joint1, stats1 = head.fit(x, y, signed, weights, 1.)
    joint10, stats10 = head.fit(x, y, signed, weights, 10.)
    del x
    prior = native.combine(np.ascontiguousarray(prior_parts), np.asarray(counts, dtype=np.int64), index)
    with np.load(SRC / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    # The fixed prior is the observed source prevalence, before the explicitly
    # class/interface-balanced training weights force that prevalence to 0.5.
    prior_gate = float(np.mean(domains)); result = {}
    gate_by_row = np.empty(len(subject)); gate_by_row[:] = np.nan; gate_by_row[test] = test_gate
    maxerr = 0.; active = 0
    for order in (ORDERS[:1] if mode == 'smoke' else ORDERS):
        perm = perms[order]
        if not np.array_equal(np.sort(perm), np.arange(len(subject))): raise ValueError('bad permutation')
        rows = perm[test[perm]]; blocks = []
        if mode == 'smoke': rows = rows[:min(61, len(rows))]
        features_out = np.empty((len(rows), 44239), np.float64)
        for sess in dict.fromkeys(session[rows]):
            loc = np.flatnonzero(session[rows] == sess); local = np.searchsorted(saved['rows'], rows[loc])
            values = native.causal(prior, np.ascontiguousarray(saved['cov'][local]),
                                   np.ascontiguousarray(saved['slow'][local]))
            check = min(5, len(local))
            prefix = native.causal(prior, np.ascontiguousarray(saved['cov'][local[:check]]),
                                   np.ascontiguousarray(saved['slow'][local[:check]]))
            if not np.allclose(values[:check], prefix, atol=1e-10, rtol=0): raise ValueError('future influence')
            features_out[loc] = values
        old = native.predict(features_out, fixed['mean'], fixed['scale'], fixed['coef'], fixed['intercept'])
        with np.load(SRC / 'runs' / ARCHIVES[order], allow_pickle=False) as z:
            archived = z[f'{order}|all_loso'][rows]
        maxerr = max(maxerr, float(np.max(np.abs(old - archived))))
        if maxerr > 1e-5 or np.any(old.argmax(1) != archived.argmax(1)):
            raise ValueError(f'original native parity failed {order}: {maxerr}')
        z = native.standardize(np.ascontiguousarray(features_out), fixed['mean'], fixed['scale'])
        g = 2 * gate_by_row[rows] - 1
        result[f'{order}|index'] = rows; result[f'{order}|gate'] = gate_by_row[rows]
        result[f'{order}|original'] = old
        result[f'{order}|shared'] = head.predict(shared, z, np.zeros(len(rows)))
        for name, model in (('joint1', joint1), ('joint10', joint10)):
            result[f'{order}|{name}'] = head.predict(model, z, g)
            result[f'{order}|{name}_equal'] = head.predict(model, z, np.zeros(len(rows)))
            result[f'{order}|{name}_prior'] = head.predict(model, z, np.full(len(rows), 2 * prior_gate - 1))
            active += int(np.sum(np.any(result[f'{order}|{name}'] != result[f'{order}|{name}_equal'], axis=1)))
    if not active: raise ValueError('gated model is a no-op')
    check_lock(root)
    result.update(shared_theta=shared, joint1_theta=joint1, joint10_theta=joint10)
    metadata = dict(held=held, fit_people=sorted(set(subject[train])), input_lock_sha256=locksha,
                    gate_fit_people=sorted(set(subject[train])), nested_gate_trace=trace,
                    gate_state=gs, oof_min=float(oof.min()), oof_max=float(oof.max()),
                    test_gate_min=float(test_gate.min()), test_gate_max=float(test_gate.max()),
                    prior_gate=prior_gate, shared=shared_stats, joint1=stats1, joint10=stats10,
                    original_max_error=maxerr, gate_changed_rows=active,
                    execution_device='cpu', cpu_affinity=affinity, cpu_threads=torch.get_num_threads(),
                    orders=list(ORDERS[:1] if mode == 'smoke' else ORDERS),
                    held_labels_opened=False, held_interface_opened=False, mode=mode,
                    seconds=time.monotonic()-started)
    result['metadata'] = json.dumps(metadata, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f: np.savez_compressed(f, **result)
    print('saved', output, 'sha256', sha(output), 'no held score', flush=True)


if __name__ == '__main__': main(int(sys.argv[1]), sys.argv[2])
