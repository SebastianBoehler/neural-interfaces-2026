"""Restore and compare the original seed-0 P10 FBCNet cohort actors."""
import ast
import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from fbl_refit_inputs import DAYS, FOLD, ORDER, PROTO, REPO, ROOT, SB, WHO, arms, device_free_trace
from returning_owner import digest, sha

PEERS = {'fb': ('fb_sbn_rot05', 2., 'fb', 35), 'fbraw': ('fbcnet_raw_sbn', 20., 'raw', 47)}


def definitions(path, names, scope):
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        key = getattr(node, 'name', None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            key = getattr(node.targets[0], 'id', None)
        if key in names:
            body.append(node)
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def source(family):
    cfg, k, stem, _ = PEERS[family]
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        d = {n: z[n] for n in ('X', 'subject', 'session', 'split')}
    assert d['X'].shape == (2655, 41, 480) and d['X'].dtype == np.float32
    d['stream'] = np.char.add(np.char.add(d['subject'], '|'), d['session'])
    import deep
    import protocol
    tr, te = {n: (tr, te) for n, tr, te in protocol.folds(d, PROTO)}[FOLD]
    fit, held = np.flatnonzero(tr), np.flatnonzero(te)
    own = np.flatnonzero(d['stream'] == WHO + '|1')
    assert (len(fit), len(held), len(own)) == (2270, 385, 192)
    assert np.isin(own, fit).all() and not np.isin(fit, held).any()
    assert set(d['stream'][held]) == {WHO + '|2', WHO + '|3'}
    with np.load(ROOT / f'runs/replay_data/feats_exact/{PROTO}-{FOLD}.npz', allow_pickle=False) as z:
        assert np.array_equal(np.sort(z['idx_tr']), fit)
    model_path = REPO / f'runs/diag/pdeep/nets/{stem}-{FOLD}-s0.pt'
    qf_path = REPO / f'runs/ttt/probs18/ttt18{ORDER}QF_{family}_base-{PROTO}-s0.npz'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    assert set(checkpoint) == {'state', 'G', 'scale'}
    t5 = definitions(REPO / 'studies/12nov/ttt5.py', {'make', 'ARCHS'}, dict(SB=SB, ARM={}))
    assert t5['ARCHS'][family] == (cfg, k)
    m = t5['make'](family, d, 0)
    m.dev = 'cpu'
    m.G, m.scale = checkpoint['G'], checkpoint['scale']
    assert not m.ref_clip and not m.ref_band and bool(m.align) == (family == 'fb')
    assert m.k == 5.0
    prior = m.C[fit].mean(0)
    assert np.array_equal(m.G, prior)
    np.testing.assert_allclose(m.scale, np.sqrt(np.trace(prior) / 41), rtol=1e-12, atol=0.)
    dims = m._aligned(fit[:2], causal=False).shape[1:]
    assert dims == (41, 480)
    m.net = deep.build(m.arch, *dims, m.pretrained, **m.net_kw).cpu()
    if m.train_fir:
        for p in m.net.spectral_filtering.parameters():
            p.requires_grad_(True)
    m._prepare()
    m.net.load_state_dict(checkpoint['state'], strict=True)
    m.net.eval()
    m.set_k(k)
    assert m.n_sbn > 0 and all(bn.k == k for bn in m.sbns())
    with np.load(qf_path, allow_pickle=False) as z:
        idx, frozen = z['index'], z['proba']
    assert idx.shape == (762,) and frozen.shape == (762, 3) and (np.diff(idx) > 0).all()
    assert np.isin(held, idx).all() and not np.isin(own, idx).any()
    inp = arms(d)
    aligned = {}
    for visit, anchor in zip(DAYS, (1238, 1326)):
        rows = inp[visit]['s281_archive']['index']
        assert rows[0] == anchor and np.array_equal(np.sort(rows), held[d['session'][held] == visit])
        aligned[visit] = np.ascontiguousarray(m._aligned(rows, causal=True))
        assert aligned[visit].dtype == np.float32 and np.isfinite(aligned[visit]).all()
        assert np.array_equal(aligned[visit][:29], m._aligned(rows[:29], causal=True))
        scaling = (d['X'][rows].astype(np.float64) / m.scale).astype(np.float32)
        if family == 'fbraw':
            assert np.array_equal(aligned[visit], scaling)
        else:
            assert np.max(np.abs(aligned[visit] - scaling)) > 1e-4
    provenance = dict(source_rows=len(fit), held_rows=len(held), own_rows=len(own), exact_prior=True,
                      source_hash=digest(fit), fit_mask_in_checkpoint=False, checkpoint=sha(model_path),
                      qf=sha(qf_path), latent_k=k, eeg_input='ea' if m.align else 'scale')
    return m, inp, aligned, idx, frozen, provenance


def difference(got, want):
    assert got.shape == want.shape and np.isfinite(got).all() and np.isfinite(want).all()
    assert np.allclose(got.sum(1), 1, atol=1e-9, rtol=0)
    return dict(max_abs=float(np.abs(got-want).max()), changed=int(np.count_nonzero(got.argmax(1) != want.argmax(1))))


def frozen_check(family, device, model, inputs, X, index, expected):
    stream = definitions(REPO / 'studies/13nov/ttt7.py', {'stream'}, dict(torch=torch, VARS={}, np=np))['stream']
    k = PEERS[family][1]
    trace = device_free_trace(SB.to_stream(model.net, k), torch.from_numpy(X['2'][:4]))
    SB.reset_stream(trace)
    buffer = io.BytesIO()
    torch.jit.save(trace, buffer)
    blob = buffer.getvalue()
    traced = torch.jit.load(io.BytesIO(blob), map_location=device).eval()
    model.net.to(device)
    def probabilities(logits):
        return torch.softmax(logits[:, :3].double(), 1).cpu().numpy()
    def sequential(x):
        SB.reset_stream(traced)
        return probabilities(torch.cat([traced(row[None]) for row in x]))
    out = {}
    with torch.no_grad():
        for visit in DAYS:
            chosen = inputs[visit]['s281_archive']
            ordered = np.searchsorted(index, chosen['index'])
            assert np.array_equal(index[ordered], chosen['index'])
            x = torch.from_numpy(X[visit]).to(device)
            eager = stream(model, x, chosen['emitted'], 0)['base']
            assert np.array_equal(eager, stream(model, x, chosen['emitted'], 0)['base'])
            SB.reset_stream(traced)
            batch = probabilities(traced(x))
            native = sequential(x)
            assert np.array_equal(native[:29], sequential(x[:29]))
            other = '3' if visit == '2' else '2'
            sequential(torch.from_numpy(X[other][:29]).to(device))
            assert np.array_equal(native[:29], sequential(x[:29]))
            ref = expected[ordered]
            out[visit] = dict(eager=difference(eager, ref), exported_batch=difference(batch, ref),
                              native=difference(native, ref), trace_eager=difference(batch, eager),
                              native_eager=difference(native, eager))
    passed = all(r['max_abs'] <= 1e-5 and r['changed'] == 0 for v in out.values() for r in v.values())
    return blob, out, passed, hashlib.sha256(blob).hexdigest()
