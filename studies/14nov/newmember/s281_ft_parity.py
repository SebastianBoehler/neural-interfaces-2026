"""One registered t=20 real-window FT implementation diagnostic; never scores labels."""
import ast
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / 'runs/sealed/final_v66_cp1c250_replay_mac/extracted'
PREFIX = ROOT / 'runs/s281_seen_relabel/within0_B2_s0.npz'


def parameter_map(eager, trace):
    """Map eager SessionBN affine parameters to traced StreamBN affine buffers."""
    ep = dict(eager.named_parameters())
    tp = dict(trace.named_parameters())
    tb = {n: b for n, b in trace.named_buffers() if n.rsplit('.', 1)[-1] in ('w', 'b')}
    mapped = {}
    for name, p in ep.items():
        pre, _, last = name.rpartition('.')
        alt = pre + '.' + {'weight': 'w', 'bias': 'b'}.get(last, last)
        target = name if name in tp else alt
        available = tp if name in tp else tb
        assert target in available, f'missing trace parameter for {name}'
        q = available[target]
        assert p.shape == q.shape, f'shape mismatch {name}'
        if p.requires_grad:
            assert target not in mapped, f'duplicate parameter {target}'
            mapped[target] = (p, q)
        else:
            assert target in tp and not q.requires_grad, f'frozen mismatch {name}'
    expected = {n for n, p in tp.items() if p.requires_grad} | set(tb)
    frozen = {n for n, p in tp.items() if not p.requires_grad}
    mapped_frozen = {n for n, p in ep.items() if not p.requires_grad}
    assert set(mapped) == expected and mapped_frozen == frozen, 'trainable/frozen parameter set mismatch'
    assert mapped, 'no trainable parameters'
    return mapped


def finite_max(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), 'nonfinite value'
    return float((a.detach() - b.detach()).abs().max())


def grad_rel(mapping, package_grads):
    assert set(mapping) == set(package_grads), 'gradient mapping mismatch'
    a, b = [], []
    for n, (p, _) in mapping.items():
        assert p.grad is not None and package_grads[n] is not None, f'missing gradient {n}'
        a.append(p.grad.detach().flatten()); b.append(package_grads[n].flatten())
    a, b = torch.cat(a), torch.cat(b)
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), 'nonfinite gradient'
    den = float(torch.linalg.vector_norm(b)); assert den > 0, 'zero gradient'
    return float(torch.linalg.vector_norm(a - b)) / den


def load_study_strong():
    path = ROOT / 'studies/13nov/ttt7.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'strong_dev']
    assert len(nodes) == 1
    scope = {'torch': torch}; exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['strong_dev']


def load_data():
    sys.path[:0] = [str(Path(__file__).parent), str(ROOT / 'experiments'), str(ROOT / 'scripts'), str(ROOT)]
    import protocol
    from sbn_trace_base import base_from_package
    from s281_seen_relabel import require_matching_artifact
    d = protocol.load(); z = np.load(PREFIX); require_matching_artifact(z, 'within0', 'B2_s0')
    idx, pos = z['index'], z['pos']; at = np.flatnonzero(idx == 1147)
    assert len(at) == 1 and pos[at[0]] == 20 and not z['recognised'][at[0]]
    stream = d['stream'][1147]; assert stream.endswith('/P10|2'), stream
    candidates = np.flatnonzero(d['stream'][idx] == stream)
    rows = candidates[np.argsort(pos[candidates])]; assert np.array_equal(pos[rows], np.arange(len(rows)))
    ordered = idx[rows]; perms = np.load(ROOT / 'runs/shuffle_perms.npz')['within|0|seen_session']
    assert np.array_equal(ordered, perms[d['stream'][perms] == stream]), 'pinned order mismatch'
    assert not z['recognised'][rows[:21]].any()
    labels = z['prefix'][rows[:20], 0]; assert labels.shape == (20, 3) and np.isfinite(labels).all()
    model, info = base_from_package(PKG, 59, d)
    assert info['trace'] == 'sbn_fblight_ea_rot05_k5_s0.pt'
    spec = importlib.util.spec_from_file_location('s281_pkgsub', PKG / 'submission.py')
    sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
    weights = np.load(PKG / 'weights.npz'); p = {k[6:]: weights[k] for k in weights.files if k.startswith('net59_')}
    inp = sub.InputStream(p)
    xp = np.stack([inp(np.nan_to_num(np.asarray(d['X'][i], np.float32))) for i in ordered[:21]])
    xd = model._aligned(ordered[:21], causal=True)
    assert np.allclose(xp, xd, atol=1e-5, rtol=1e-5), 'target session input mismatch'
    return model.net.eval(), torch.from_numpy(np.ascontiguousarray(xp)), labels, info


def run():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    sys.path.insert(0, str(ROOT))
    from lean.src66 import finetune as ft
    import zipfile, hashlib
    archive = ROOT / 'runs/sealed/final_zips/v66_cp1_cap250.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed'
    with zipfile.ZipFile(archive) as z:
        source = z.read('submission.py').decode()
    def cls(src):
        return ast.dump(next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == 'FineTune'), include_attributes=False)
    assert cls(source) == cls((ROOT / 'lean/src66/finetune.py').read_text()), 'package FineTune source mismatch'
    eager, X, P, info = load_data(); trace = torch.jit.load(str(PKG / info['trace']), map_location='cpu').eval()
    mapping = parameter_map(eager, trace); tracked = {id(q): n for n, (_, q) in mapping.items()}
    for _, q in mapping.values(): q.requires_grad_(True)
    assert all(p.dtype == torch.float32 and q.dtype == torch.float32 for p, q in mapping.values())
    labels, weights = P.argmax(1), P.max(1).astype(np.float32)
    initial = {n: q.detach().clone() for n, (_, q) in mapping.items()}
    ft.FineTune.zero(trace)
    with torch.no_grad():
        a = eager(X[:20])[:, :3]; b = trace(X[:20])[:, :3]
    initial_logits = finite_max(a, b); assert initial_logits <= 1e-4, initial_logits
    captures, augments, loss_terms = [], [], []
    Adam = torch.optim.Adam
    class RecordingAdam(Adam):
        def step(self, *args, **kwargs):
            grads = {tracked[id(q)]: None if q.grad is None else q.grad.detach().clone() for g in self.param_groups for q in g['params']}
            result = super().step(*args, **kwargs)
            captures.append((grads, {n: q.detach().clone() for n, (_, q) in mapping.items()}))
            return result
    strong = ft.strong
    def recorded_strong(x, gen):
        result = strong(x, gen); augments.append((result.detach().clone(), gen.get_state().clone())); return result
    ce0, kl0 = F.cross_entropy, F.kl_div
    def ce(*args, **kwargs):
        result = ce0(*args, **kwargs); loss_terms.append(('ce', result.detach().clone())); return result
    def kl(*args, **kwargs):
        result = kl0(*args, **kwargs); loss_terms.append(('kl', result.detach().clone())); return result
    obj = ft.FineTune.__new__(ft.FineTune); obj.nets = {'fbl': trace}; obj.X = {'fbl': [x.numpy() for x in X[:20]]}
    obj.dev = 'cpu'; obj.lr = .001; obj.lam = 1.; obj.n_steps = 5; obj.gens = {'fbl': torch.Generator().manual_seed(0)}
    obj.theta0 = {'fbl': [q.detach().clone() for q in ft.FineTune.params(trace)]}; obj.steps = obj.refits = 0; obj.dtheta = 0.
    ft.FineTune.zero(trace)
    with patch.object(torch.optim, 'Adam', RecordingAdam), patch.object(ft, 'strong', recorded_strong), patch.object(F, 'cross_entropy', ce), patch.object(F, 'kl_div', kl), torch.jit.optimized_execution(False):
        obj._refit(20, labels, weights)
    assert obj.steps == 5 and obj.refits == 1 and obj.dtheta > 0 and len(captures) == len(augments) == 5
    with torch.no_grad(): package_next = torch.softmax(trace(X[20:21])[:, :3].double(), 1)
    generator = torch.Generator().manual_seed(0); study_strong = load_study_strong()
    opt = torch.optim.AdamW(list(eager.parameters()), lr=.001, weight_decay=0.)
    lab = torch.as_tensor(labels); wt = torch.as_tensor(weights); reports = []
    for step in range(5):
        opt.zero_grad(); clean = eager(X[:20])[:, :3]
        augmented = study_strong(X[:20], generator)
        assert torch.equal(augmented, augments[step][0]) and torch.equal(generator.get_state(), augments[step][1]), 'augmentation/RNG mismatch'
        za = eager(augmented)[:, :3]
        loss = (wt * F.cross_entropy(za, lab, reduction='none')).sum() / wt.sum().clamp_min(1e-6) + F.kl_div(F.log_softmax(za, 1), F.softmax(clean.detach(), 1), reduction='batchmean')
        pk_loss = (wt * loss_terms[2 * step][1]).sum() / wt.sum().clamp_min(1e-6) + loss_terms[2 * step + 1][1]
        ld = finite_max(loss, pk_loss); assert ld <= 1e-4, ('loss', step, ld)
        loss.backward(); gd = grad_rel(mapping, captures[step][0]); assert gd <= 1e-3, ('gradient', step, gd)
        opt.step(); pd = max(finite_max(p, captures[step][1][n]) for n, (p, _) in mapping.items())
        assert pd <= 1e-4, ('parameter', step, pd)
        assert max(float((p.detach() - initial[n]).abs().max()) for n, (p, _) in mapping.items()) > 0, 'zero update'
        reports.append(dict(step=step + 1, loss_diff=ld, gradient_relative_l2=gd, parameter_max_diff=pd))
    with torch.no_grad(): eager_next = torch.softmax(eager(X[:21])[20:21, :3].double(), 1)
    probdiff = finite_max(eager_next, package_next); assert probdiff <= 1e-4, ('next probability', probdiff)
    print(dict(model=info, initial_logits_diff=initial_logits, steps=reports, next_probability_diff=probdiff, nonzero_update=obj.dtheta), flush=True)
    print('S281 FT PARITY PASS', flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 1, 'this diagnostic has no runtime options'
    run()
