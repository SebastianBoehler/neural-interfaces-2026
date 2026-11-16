"""Fail-closed source, nested-gate and complete output validation."""
import json
import numpy as np

MODES = ('original', 'shared', 'joint1', 'joint10', 'joint1_equal', 'joint10_equal',
         'joint1_prior', 'joint10_prior')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def trace_check(meta, source_people, held):
    people = set(source_people)
    if (len(people) != 9 or held in people or set(meta['fit_people']) != people or
            set(meta['gate_fit_people']) != people or set(meta['nested_gate_trace']) != people):
        raise ValueError('outer fold identity mismatch')
    for person, entry in meta['nested_gate_trace'].items():
        rest = people - {person}
        if (set(entry['gate_fit_people']) != rest or set(entry['calibration_people']) != rest or
                set(entry['inner_fits']) != rest):
            raise ValueError('gate person leaked into own outer calibration')
        for other, fit in entry['inner_fits'].items():
            if set(fit) != rest - {other}: raise ValueError('inner calibration fold leaked')
    if (meta['held_labels_opened'] is not False or meta['held_interface_opened'] is not False or
            meta['execution_device'] != 'cpu' or meta['cpu_threads'] != 6 or len(set(meta['cpu_affinity'])) != 6 or
            meta['original_max_error'] > 1e-5 or meta['gate_changed_rows'] <= 0 or
            meta['test_gate_max'] - meta['test_gate_min'] <= 1e-8 or
            meta['oof_max'] - meta['oof_min'] <= 1e-8):
        raise ValueError('inactive, leaked or unmatched output')
    for mode in ('shared', 'joint1', 'joint10'):
        stats = meta[mode]
        if not np.isfinite(stats['max_gradient']) or stats['max_gradient'] > 1e-5:
            raise ValueError('unconverged head')
        if mode != 'shared' and stats['deviation_norm'] <= 1e-10:
            raise ValueError('inactive expert')


def load_fold(path, held, subject, session, perms, lock_sha, mode):
    held_id = f'Dreyer2026Proteus/{held}'; test = subject == held_id
    source_people = set(subject[~test]); result = {}
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata'].item()))
        if (meta['held'] != held or meta['mode'] != mode or meta['input_lock_sha256'] != lock_sha or
                meta['orders'] != list(ORDERS[:1] if mode == 'smoke' else ORDERS)):
            raise ValueError('fold provenance mismatch')
        trace_check(meta, source_people, held_id)
        for theta in ('shared_theta', 'joint1_theta', 'joint10_theta'):
            if z[theta].shape != (2, 3, 44240) or not np.isfinite(z[theta]).all():
                raise ValueError('invalid fitted head')
        if np.any(z['shared_theta'][1]): raise ValueError('shared head has expert weights')
        for order in meta['orders']:
            rows = perms[order][test[perms[order]]]
            if mode == 'smoke': rows = rows[:min(61, len(rows))]
            if not np.array_equal(z[f'{order}|index'], rows): raise ValueError('wrong delivery rows')
            g = z[f'{order}|gate']
            if g.shape != (len(rows),) or not np.isfinite(g).all() or np.any((g < 0) | (g > 1)):
                raise ValueError('invalid gate output')
            for name in MODES:
                p = z[f'{order}|{name}']
                if (p.shape != (len(rows), 3) or not np.isfinite(p).all() or
                        np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(1), 1, atol=1e-12, rtol=0)):
                    raise ValueError(f'invalid {order}/{name} probabilities')
                result[order, name] = (rows.copy(), p.copy())
    return meta, result
