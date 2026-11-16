"""Registered within0 P04 held-session stage parity; no scores or predictions."""
import ast
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'replay'), str(ROOT / 'studies/12nov'), str(ROOT)]
from lean.src66.numerics import softmax
from lean.src66.personal import OwnStage
from lean.src66.routing import Route
from lean.src66.stages import TSEM

ZIP_SHA = '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed'
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
GATE_SHA = 'cf581194de3df6352d7bd118300446d12bd5c1a471ec1e0ce73cd6acba04d7ea'
PERMS_SHA = '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'
FOLD_SHA = '0bdeaecdf7d8f5a4ee9b357fab1456d4da08eea221a507b376402b36c0eb0b82'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def source_node(text, name):
    nodes = [n for n in ast.parse(text).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
    assert len(nodes) == 1, f'missing/duplicate source {name}'
    return ast.dump(nodes[0], include_attributes=False)


def diff(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), f'{what}: shape/nonfinite'
    maximum = float(np.max(np.abs(a - b))); assert maximum <= 1e-8, f'{what}: {maximum} > 1e-8'
    return maximum


def arrays(personal, own):
    scaler, lr = personal.T.steps[0][1], personal.T.steps[1][1]
    out = dict(prior=np.stack(personal.prior), csp=np.stack(personal.W), t_mu=scaler.mean_, t_sd=scaler.scale_,
               t_W=lr.coef_, t_c=lr.intercept_, c_W=personal.C.coef_, c_c=personal.C.intercept_,
               s_W=personal.S.coef_, s_c=personal.S.intercept_)
    assert all(np.isfinite(v).all() for v in out.values()) and (out['t_sd'] > 0).all()
    z = {f'own_{k}': np.stack([v]).astype(np.float64) for k, v in out.items()}
    z.update(own_bands=np.asarray(own.BANDS, np.float64), own_beta=np.float64(.10), own_kref=np.float64(own.KREF))
    return z


class Gate:
    value = None
    def step(self, *args): return self.value


class OwnOff:
    cur = None
    def observe(self, x): return False


class CaptureTail:
    def step(self, x, u, p): return p.copy()


def head_route(head, U, ramps):
    """Use actual Route.step and algebraically recover its injected head posterior."""
    assert len(U) == len(ramps) and np.isfinite(U).all() and np.isfinite(ramps).all()
    assert ((ramps >= 0) & (ramps <= 1)).all()
    route = Route.__new__(Route)
    route.W, route.c = head.coef_[None], head.intercept_[None]
    route.g, route.a, route.n_tsem = 3., 1/16, 128
    route.U, route.Q, route.last = [], [], {}
    route.contrast, route.identity, route.own, route.tail = Gate(), Gate(), OwnOff(), CaptureTail()
    route.pri, route.kappa = None, 0.
    route.log = dict(n=0, routed=0, degen=0, staged=0, err=0, first='')
    mix, core = np.array([.2, .3, .5]), np.array([.3, .4, .3]); output = []
    for u, r in zip(U, ramps):
        route.contrast.value = (0, 0., float(r)) if r > 0 else None
        p = route.step(None, u, mix, core, np.array([1/3] * 3))
        if r > 0:
            output.append((p * (1 - r * route.a + route.g * r * route.a) - mix + r * route.a * core) / (route.g * r * route.a))
    route.log['n'] = 0  # suppress destructor report for this partial standalone object
    return np.asarray(output)


def main():
    data_path = ROOT/'data/proteus_windows.npz'; gate_path = ROOT/'runs/s281_seen_gate_trace/within0.npz'
    perm_path = ROOT/'runs/shuffle_perms.npz'; zip_path = ROOT/'runs/sealed/final_zips/v66_cp1_cap250.zip'
    fold_path = ROOT/'runs/replay_data/feats_exact/seen_session-P04_s23.npz'
    assert sha(data_path) == DATA_SHA and sha(fold_path) == FOLD_SHA and sha(gate_path) == GATE_SHA and sha(perm_path) == PERMS_SHA and sha(zip_path) == ZIP_SHA
    import ownspace as O
    import rlib3 as R3
    import route as RR
    from sklearn.linear_model import LogisticRegression
    with zipfile.ZipFile(zip_path) as z:
        source = z.read('submission.py').decode()
        with np.load(io.BytesIO(z.read('weights.npz'))) as w:
            assert float(w['own_beta']) == .10 and float(w['own_kref']) == O.KREF and float(w['v6_tpri_kappa']) == 20.
            stage_keys = ('tsem_lam', 'tsem_rho', 'tsem_shrink', 'tsem_m0', 'tsem_n0', 'tsem_nmin', 'tsem_boost', 'eb_T')
            stage_params = {key: w[key].copy() for key in stage_keys}
    for name, path in (('OwnStage','lean/src66/personal.py'), ('Route','lean/src66/routing.py'),
                       ('TSEM','lean/src66/stages.py'), ('softmax','lean/src66/numerics.py')):
        assert source_node(source, name) == source_node((ROOT/path).read_text(), name), f'package class differs: {name}'
    with np.load(data_path) as d, np.load(fold_path) as f, np.load(perm_path) as pp, np.load(gate_path) as g:
        idx_tr = f['idx_tr'].astype(int); held = np.flatnonzero((d['subject'] == 'Dreyer2026Proteus/P04') & (d['session'] != '1'))
        assert len(np.unique(idx_tr)) == len(idx_tr) and not np.intersect1d(idx_tr, held).size
        assert np.array_equal(np.sort(idx_tr), np.setdiff1d(np.arange(len(d['y'])), held)), 'fold training rows mismatch'
        assert set(d['session'][held]) == {'2','3'}
        own = idx_tr[d['subject'][idx_tr] == 'Dreyer2026Proteus/P04']
        assert len(own) and set(d['session'][own]) == {'1'} and not np.intersect1d(own, held).size
        train_pc = f['pc_tr'][d['subject'][idx_tr] == 'Dreyer2026Proteus/P04']
        train_y = f['y_tr'][d['subject'][idx_tr] == 'Dreyer2026Proteus/P04']
        assert np.array_equal(train_y, d['y'][own]) and set(train_y.tolist()) == {0,1,2}
        assert list(f['sessions']).count('Dreyer2026Proteus/P04|2') == 1
        k = list(f['sessions']).index('Dreyer2026Proteus/P04|2'); original, pc = f[f'{k}|idx'], f[f'{k}|pc']
        session = pp['within|0|seen_session']; rows = session[np.isin(session, original)]
        assert set(rows.tolist()) == set(original.tolist()) and len(rows) == len(original)
        lut = {int(v): i for i, v in enumerate(original)}; U = pc[[lut[int(v)] for v in rows]].astype(np.float64)
        gi = g['index']; ix = np.searchsorted(gi, rows); assert np.array_equal(gi[ix], rows)
        ramp = g['B2_s0|winner_r'][ix]; person = g['B2_s0|winner_person'][ix]
        assert np.any(ramp > 0) and np.all(person[ramp > 0] == 'P04')
        X = d['X'][rows].copy(); assert np.array_equal(O.y, d['y']) and np.array_equal(O.X[rows], X.astype(np.float64))
    head = R3.ps1_heads({'P04': [(train_pc, train_y)]}, ['P04'], C=.1)['P04']
    assert isinstance(head, LogisticRegression) and np.array_equal(head.classes_, np.arange(3))
    expected_head = head.predict_proba(RR.causal_std(U[:, :256])); forced = head_route(head, U, np.ones(len(U)))
    actual = head_route(head, U, ramp); d_head = diff(forced, expected_head, 'head all windows')
    d_gate = diff(actual, expected_head[ramp > 0], 'head gated windows')
    sd = R3.stdz(train_pc[:, :128]); prior = np.stack([sd[train_y == c].mean(0) for c in range(3)])
    other = (train_pc[:, :128] - train_pc[:, :128].mean(0)) / np.maximum(train_pc[:, :128].std(0), 1e-6)
    reference_prior = np.stack([other[train_y == c].mean(0) for c in range(3)])
    d_prior = diff(prior, reference_prior, 'prior')
    # TSEM takes this prior per recognised window; no public-person package arrays are loaded.
    ts = TSEM(stage_params)
    ts.prior = (20. * float(ramp[ramp > 0][0]), prior); assert np.array_equal(ts.prior[1], reference_prior)
    personal = O.Personal([own]); stage = OwnStage(arrays(personal, O)); stage.reset_state()
    raw = personal.stream(rows); reference = sum(np.log(np.clip(raw[k], 1e-3, 1)) for k in 'TCS') / 3
    got = []
    for x in X:
        assert stage.observe(x), 'degenerate window'
        got.append(stage.log_members(0))
    d_own = diff(np.asarray(got), reference, 'own-space logs')
    result = dict(fold='P04_s23', session='P04|2', order='within0', base='B2_s0', train_rows=len(idx_tr),
                  own_train_rows=len(own), held_rows=len(held), windows=len(rows), routed_windows=int((ramp > 0).sum()),
                  head_max_diff=d_head, gated_head_max_diff=d_gate, prior_max_diff=d_prior, own_log_max_diff=d_own,
                  input_sha256={'data': sha(data_path), 'fold': sha(fold_path), 'gate': sha(gate_path), 'perms': sha(perm_path), 'zip': sha(zip_path)},
                  train_rows_sha256=hashlib.sha256(np.ascontiguousarray(idx_tr, dtype=np.int64).tobytes()).hexdigest(),
                  own_rows_sha256=hashlib.sha256(np.ascontiguousarray(own, dtype=np.int64).tobytes()).hexdigest(),
                  held_rows_sha256=hashlib.sha256(np.ascontiguousarray(held, dtype=np.int64).tobytes()).hexdigest(),
                  source_sha256={p: sha(ROOT/p) for p in ('studies/14nov/newmember/s281_fold_stage_diag.py','studies/12nov/ownspace.py','replay/rlib3.py','replay/route.py','lean/src66/personal.py','lean/src66/routing.py','lean/src66/stages.py')})
    print(json.dumps(result, sort_keys=True), flush=True)
    print('FOLD STAGE PARITY PASS: no treatment score', flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 1, 'no runtime options'
    main()
