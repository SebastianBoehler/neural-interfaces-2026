"""Authenticate and assemble one P10 seed-0 FBLight trace and four input arms."""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

from returning_owner import BASE, ROOT, ORDER, PROTO, STACK, digest, sha

REPO = BASE / 'neural-interfaces-2026'
S281 = BASE / 'ni26_s281_seen_prepare_20261116/full'
BRIDGE = BASE / 'ni26_friday_ea_returning_bridge_v2_20261118'
PACKAGE = S281 / 'runs/sealed/final_zips/v66_cp1_cap250.zip'
QF = REPO / f'runs/ttt/probs18/ttt18{ORDER}QF_fbl_base-{PROTO}-s0.npz'
CKPT = REPO / 'runs/diag/pdeep/nets/fbl-P10_s23-s0.pt'
WHO, FOLD = 'Dreyer2026Proteus/P10', 'P10_s23'
DAYS = ('2', '3')
ARMS = ('s281_archive', 'bridge_archive', 'fresh_recording', 'fresh_causal')

sys.path[:0] = [str(REPO / p) for p in ('experiments', 'scripts')]
import session_bn as SB
from export_external_teacher import device_free_trace
import s281_seen_fbl as s281
import s281_seen_fbl_checkpoint as check


def source():
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        d = {k: z[k] for k in ('X', 'subject', 'session', 'split')}
    assert d['X'].shape == (2655, 41, 480) and d['X'].dtype == np.float32
    d['stream'] = np.char.add(np.char.add(d['subject'], '|'), d['session'])
    check.ROOT = REPO
    model, held = check.load_fold(d, FOLD, 0)
    fit, test = np.flatnonzero(~held), np.flatnonzero(held)
    own = np.flatnonzero(d['stream'] == WHO + '|1')
    with np.load(ROOT / f'runs/replay_data/feats_exact/{PROTO}-{FOLD}.npz', allow_pickle=False) as z:
        assert np.array_equal(np.sort(z['idx_tr']), fit)
    assert (len(fit), len(test), len(own)) == (2270, 385, 192)
    assert np.isin(own, fit).all() and not np.isin(fit, test).any()
    archive = np.load(QF, allow_pickle=False)
    assert np.isin(test, archive['index']).all() and len(archive['index']) == 762
    assert model.dev == 'cpu' and model.n_sbn > 0 and model.k == 5.0
    assert not model.ref_band and not model.ref_clip
    prior = model.C[fit].mean(0)
    error = float(np.max(np.abs(prior - model.G)))
    np.testing.assert_allclose(model.G, prior, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(model.scale, np.sqrt(np.trace(prior) / len(prior)), rtol=1e-12, atol=0.)
    stats = dict(fit_rows=len(fit), held_rows=len(test), own_rows=len(own), max_prior_error=error,
                 source_mask_sha256=digest(fit), checkpoint_sha256=sha(CKPT), qf_sha256=sha(QF),
                 checkpoint_contains_fit_mask=False)
    return d, model, archive, stats


def authenticate(path, name):
    proof = json.loads((path / (name + '.json')).read_text())
    file = path / (name + '.npz')
    assert proof['status'] == 'passed' and not proof['scored'] and proof['label_free_targets']
    assert proof['bank'] == sha(file)
    return np.load(file, allow_pickle=False), proof


def arms(d):
    s281.ROOT = S281
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        delivery = z['full|0|seen_session']
    zero, zp = authenticate(BRIDGE, 'zero')
    fresh, fp = authenticate(BRIDGE, 'matched')
    assert np.array_equal(zero['index'], fresh['index'])
    assert np.array_equal(delivery, zero['delivery']) and np.array_equal(delivery, fresh['delivery'])
    routes = {'bridge_archive': zp['stages']['route'],
              **{a: fp['stages'][a]['route'] for a in ARMS[2:]}}
    inputs = {}
    for visit, first, age in zip(DAYS, (1238, 1326), (28, 10)):
        name = WHO + '|' + visit
        saved = s281.session_inputs(ORDER, STACK, name, d)
        rows = delivery[d['stream'][delivery] == name]
        assert rows[0] == first and np.array_equal(saved['index'], rows)
        assert np.flatnonzero(saved['recognised'])[0] == age
        inputs[visit] = {'s281_archive': saved}
        for arm in ARMS[1:]:
            bank, tag = (zero, '') if arm == 'bridge_archive' else (fresh, arm + '|')
            prefix = tag + 'P10-' + visit + '|'
            assert routes[arm][name] == {'owner': 'P10', 'age': age}
            assert np.array_equal(rows, bank[prefix + 'rows'])
            i = np.searchsorted(bank['index'], rows)
            assert np.array_equal(bank['index'][i], rows)
            inputs[visit][arm] = dict(index=rows, emitted=bank[tag + 'shadow'][i],
                                      prefix=bank[prefix + 'refits'], grid=bank[prefix + 'ages'],
                                      recognised=np.arange(len(rows)) >= age)
    zero.close()
    fresh.close()
    return inputs


def compare(a, b, tol=1e-5):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    err = float(np.abs(a - b).max())
    assert err <= tol and np.array_equal(a.argmax(1), b.argmax(1)), err
    return err


def export(d, model, qf, inputs):
    x = {}
    for visit in DAYS:
        rows = inputs[visit]['s281_archive']['index']
        assert (d['stream'][rows] == WHO + '|' + visit).all()
        x[visit] = np.ascontiguousarray(model._aligned(rows, causal=True), dtype=np.float32)
        assert np.array_equal(x[visit][:29], model._aligned(rows[:29], causal=True))
    trace = device_free_trace(SB.to_stream(model.net, 5.), torch.from_numpy(x['2'][:4]))
    SB.reset_stream(trace)
    buffer = io.BytesIO()
    torch.jit.save(trace, buffer)
    blob = buffer.getvalue()
    traced = torch.jit.load(io.BytesIO(blob), map_location='cpu').eval()
    result = {}
    idx, frozen = qf['index'], qf['proba']
    def predict(tensor):
        return torch.softmax(tensor[:, :3].double(), 1).numpy()
    def sequential(tensor):
        SB.reset_stream(traced)
        return predict(torch.cat([traced(row[None]) for row in tensor]))
    with torch.no_grad():
        for visit in DAYS:
            rows = inputs[visit]['s281_archive']['index']
            where = np.searchsorted(idx, rows)
            assert np.array_equal(idx[where], rows)
            tensor = torch.from_numpy(x[visit])
            eager = predict(model.net(tensor))
            SB.reset_stream(traced)
            batch = predict(traced(tensor))
            native = sequential(tensor)
            result[visit] = dict(eager_qf=compare(eager, frozen[where]), trace_eager=compare(batch, eager),
                                 sequential_qf=compare(native, frozen[where]), sequential_eager=compare(native, eager))
            assert np.array_equal(native[:29], sequential(tensor[:29]))
            tensor_other = torch.from_numpy(x['3' if visit == '2' else '2'])
            sequential(tensor_other[:29])
            assert np.array_equal(native[:29], sequential(tensor[:29]))
            result[visit]['frozen_prefix_and_reset_exact'] = True
    return x, blob, result, hashlib.sha256(blob).hexdigest()
