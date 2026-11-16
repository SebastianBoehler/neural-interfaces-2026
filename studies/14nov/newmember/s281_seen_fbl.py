"""Fail-closed FBLight FineTune consumer and shipped-model implementation smoke.

The shipped trace was trained on public participants. Its outputs are diagnostics,
not held-out predictions. No unverified dev trace is accepted by the CLI.
"""
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT / 'experiments'), str(ROOT)]
from s281_seen_relabel import (BASES, CAP, ORDERS, PACKAGE_SHA, PERMS_SHA, require_matching_artifact,
                                require_matching_base, validate_prefix)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_inputs(order, base, session, d):
    """Validate matching immutable sources; return one session in pinned delivery order."""
    assert order in ORDERS and base in BASES, 'invalid order or base'
    path = ROOT / f'runs/s281_seen_relabel/{order}_{base}.npz'
    trace_path = ROOT / f'runs/s281_seen_route/{order}.npz'
    with np.load(path) as q, np.load(trace_path) as trace:
        require_matching_artifact(q, order, base); require_matching_base(trace, base)
        assert str(q['package_sha256']) == PACKAGE_SHA and str(q['trace_sha256']) == sha(trace_path), 'provenance mismatch'
        idx, pos, rec = q['index'], q['pos'], q['recognised']
        assert str(trace['order']) == order and np.array_equal(trace['index'], idx), 'route index/order mismatch'
        assert np.array_equal(rec, trace[f'{base}|recognised']), 'recognition trace mismatch'
        assert len(np.unique(idx)) == len(idx) and idx.min() >= 0 and idx.max() < len(d['stream'])
        assert q['emitted'].shape == (len(idx), 3) and np.isfinite(q['emitted']).all()
        assert np.all((0 <= q['emitted']) & (q['emitted'] <= 1)) and np.allclose(q['emitted'].sum(1), 1)
        assert np.allclose(q['emitted'][~rec], trace[f'{base}|shadow'][~rec], rtol=0, atol=1e-9), 'label shadow mismatch'
        sessions = d['stream'][idx]
        validate_prefix(q['prefix'], pos, sessions, q['grid'], rec)
        perm_path = ROOT / 'runs/shuffle_perms.npz'; assert sha(perm_path) == PERMS_SHA
        scheme = order.rstrip('0123456789'); seed = order[len(scheme):]
        with np.load(perm_path) as perms:
            order_rows = perms[f'{scheme}|{int(seed)}|seen_session']
        assert len(order_rows) == len(idx) and np.array_equal(np.sort(order_rows), np.sort(idx)), 'pinned rows mismatch'
        rows = np.flatnonzero(sessions == session); assert len(rows), 'unknown session'
        rows = rows[np.argsort(pos[rows])]; global_rows = idx[rows]
        assert np.array_equal(global_rows, order_rows[d['stream'][order_rows] == session]), 'session order mismatch'
        return dict(index=global_rows.copy(), prefix=q['prefix'][rows].copy(), grid=q['grid'].copy(),
                    emitted=q['emitted'][rows].copy(), recognised=rec[rows].copy())


def run_session(ft, inputs, record):
    """Run one fresh FineTune session. NaN marks no FT replacement, including recognised windows."""
    n = len(inputs); rec = record['recognised']; prefix = record['prefix']; grid = record['grid']
    assert n == len(record['index']) == len(rec) == len(record['emitted']) == len(prefix)
    assert inputs.shape == (n, 41, 480) and inputs.dtype == np.float32 and np.isfinite(inputs).all(), 'invalid FBLight inputs'
    assert len(ft.lab) == 0 and ft.nets is None, 'FineTune must start fresh for every session'
    assert np.all(rec[1:] >= rec[:-1]), 'recognition unlatched'
    pred = np.full((n, 3), np.nan); refits = []
    for t, x in enumerate(inputs):
        if rec[t]:
            break
        def relabel(k):
            assert k == t and k <= CAP and k in grid, 'unexpected refit time'
            P = prefix[:k, int((k - 20) // 10)]
            assert P.shape == (k, 3) and np.isfinite(P).all(), 'invalid refit prefix'
            assert np.all((0 <= P) & (P <= 1)) and np.allclose(P.sum(axis=1), 1), 'invalid posterior prefix'
            refits.append(k); return P
        output = ft.step({}, {59: x}, record['emitted'][t], relabel)
        assert set(output) <= {'fbl'}, 'unexpected FT family'
        if 'fbl' in output:
            assert t >= 20 and np.isfinite(output['fbl']).all(), 'invalid FT output'
            pred[t] = output['fbl']
    return pred, refits


def shipped_ft():
    """Construct exactly the shipped FBLight family; no dev-fold trace is assumed."""
    from lean.src66.finetune import FineTune
    import ast
    pkg = ROOT / 'runs/sealed/final_v66_cp1c250_replay_mac/extracted'
    bundle = ROOT / 'runs/sealed/final_zips/v66_cp1_cap250.zip'
    assert sha(bundle) == PACKAGE_SHA, 'package hash mismatch'
    with zipfile.ZipFile(bundle) as z:
        source = z.read('submission.py').decode(); wb = z.read('weights.npz')
        weights = dict(np.load(io.BytesIO(wb))); member_file = str(weights['net59_sjit_file'])
        assert (pkg / member_file).read_bytes() == z.read(member_file), 'shipped trace mismatch'
    assert member_file == 'sbn_fblight_ea_rot05_k5_s0.pt'
    def get_cls(s):
        return ast.dump(next(n for n in ast.parse(s).body if isinstance(n, ast.ClassDef) and n.name == 'FineTune'), include_attributes=False)
    assert get_cls(source) == get_cls((ROOT / 'lean/src66/finetune.py').read_text()), 'FineTune source mismatch'
    required = {'v6_ft_on': 1, 'v6_ft_relabel': 1, 'v6_ft_labels': 0, 'v6_routed': 1, 'v6_ft_cap': 250,
                'v6_ft_t0': 20, 'v6_ft_every': 10, 'v6_ft_steps': 5, 'v6_ft_seed': 0, 'v6_ft_lr': .001, 'v6_ft_lam': 1.}
    for key, value in required.items(): assert weights[key].item() == value, f'package setting mismatch {key}'
    names = list(weights['v6_slots']); members = list(weights['v6_ft_member'])
    assert int(members[names.index('fbl')]) == 59
    weights['v6_slots'] = np.array(['fbl']); weights['v6_ft_member'] = np.array([59])
    w = weights['member_weights'].astype(np.float64); w /= w.sum()
    return FineTune(weights, w, 'cpu', pkg), weights


def smoke():
    import torch
    import protocol
    from lean.src66.inputs import InputStream
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    d = protocol.load(); session = 'Dreyer2026Proteus/P10|2'
    record = session_inputs('within0', 'B2_s0', session, d); ft, weights = shipped_ft()
    first = np.flatnonzero(record['recognised']); assert len(first) and first[0] == 60
    p = {k[6:]: weights[k] for k in weights if k.startswith('net59_')}
    stream = InputStream(p)
    rows = record['index']; X = np.stack([stream(np.nan_to_num(np.asarray(d['X'][r], np.float32))) for r in rows])
    pred, refits = run_session(ft, X, record)
    assert refits == [20, 30, 40, 50] and ft.refits == 4 and ft.relabels == 4 and ft.dtheta > 0
    assert np.isnan(pred[:20]).all() and np.isfinite(pred[20:60]).all() and np.isnan(pred[60:]).all()
    print(f'P10 session 2: recognised_at=60 refits={refits} relabels={ft.relabels} max_update={ft.dtheta:.9g}; shipped-model implementation smoke only', flush=True)
    print('S281 SEEN FBL SMOKE PASS', flush=True)


if __name__ == '__main__':
    assert sys.argv[1:] == ['smoke'], 'only the shipped-model implementation smoke is available'
    smoke()
