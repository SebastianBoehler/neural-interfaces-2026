"""Run the shipped FBLight age-20 adaptation on authenticated no-gold inputs."""
import ast
import copy
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from fbl_refit_inputs import ARMS, DAYS, PACKAGE, arms, export, source
from returning_owner import digest, sha
from s281_seen_fbl import run_session
from s281_seen_relabel import PACKAGE_SHA


def shipped():
    assert sha(PACKAGE) == PACKAGE_SHA
    with zipfile.ZipFile(PACKAGE) as archive:
        syntax = ast.parse(archive.read('submission.py'))
        nodes = [n for n in syntax.body if getattr(n, 'name', '') in ('FineTune', 'strong') or
                 (isinstance(n, ast.Assign) and any(isinstance(k, ast.Name) and k.id == 'STATE' for k in n.targets))]
        assert len(nodes) == 3
        with np.load(io.BytesIO(archive.read('weights.npz')), allow_pickle=False) as z:
            weights = {k: z[k] for k in z.files}
        file = str(weights['net59_sjit_file'])
        assert file == 'sbn_fblight_ea_rot05_k5_s0.pt'
        with Path(file).open('xb') as handle:
            handle.write(archive.read(file))
    scope = dict(copy=copy, io=io, np=np, torch=torch, F=F, time=time, Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PACKAGE) + '/submission.py', 'exec'), scope)
    assert weights['v6_ft_member'][list(weights['v6_slots']).index('fbl')] == 59
    keys = dict(on=1, relabel=1, labels=0, cap=250, t0=20, every=10, steps=5, seed=0, lr=.001, lam=1.)
    assert all(weights['v6_ft_' + k].item() == v for k, v in keys.items())
    assert int(weights['v6_routed']) == 1
    weights.update(v6_slots=np.array(['fbl']), v6_ft_member=np.array([59]))
    w = weights['member_weights'].astype(np.float64)
    return scope['FineTune'](weights, w / w.sum(), 'cpu', Path('.')), scope


def consume(template, scope, aligned, data):
    ft = copy.deepcopy(template)
    strong = scope['strong']
    calls, augmentations = [], []
    actual_refit = ft._refit
    boundary = int(np.flatnonzero(data['recognised'])[0])
    def observe_aug(x, gen):
        assert boundary == 28 and x.shape == (20, 41, 480)
        assert np.array_equal(x.numpy(), aligned[:20])
        y = strong(x, gen)
        assert not torch.equal(x, y) and torch.isfinite(y).all()
        augmentations.append(digest(y.numpy()))
        return y
    def observe_refit(t, lab, wt):
        assert t == 20 and boundary == 28 and not calls
        p = data['prefix'][:t, 0]
        assert np.array_equal(lab, p.argmax(1)) and np.array_equal(wt, p.max(1))
        assert len(ft.lab) == len(ft.wt) == t
        actual_refit(t, lab, wt)
        fitted = torch.cat([p.detach().flatten() for p in ft.params(ft.nets['fbl'])]).numpy()
        assert np.isfinite(fitted).all()
        calls.append(dict(age=t, labels=lab.tolist(), weights=wt.astype(np.float32).tolist(),
                          updated_parameters=digest(fitted), rng=digest(ft.gens['fbl'].get_state().numpy())))
    scope['strong'], ft._refit = observe_aug, observe_refit
    try:
        prob, ticks = run_session(ft, aligned, data)
    finally:
        scope['strong'] = strong
    active = boundary == 28
    assert ticks == ([20] if active else []) and len(ft.lab) == len(ft.wt) == boundary
    assert ft.capped == ft.over == 0 and ft.refits == ft.relabels == int(active) and (ft.dtheta > 0) == active
    want = (np.arange(len(prob)) >= 20) & (np.arange(len(prob)) < boundary)
    assert np.array_equal(np.isfinite(prob).all(1), want) and np.isnan(prob[~want]).all()
    assert ft.steps == len(augmentations) == (5 if active else 0)
    return prob, dict(recognition_age=boundary, active_rows=int(want.sum()), refits=ticks,
                      steps=ft.steps, update=ft.dtheta, optimiser=calls, augmentations=augmentations)


def delta(a, b):
    keep = np.isfinite(a).all(1)
    assert np.array_equal(keep, np.isfinite(b).all(1))
    x, y = a[keep], b[keep]
    return dict(max_abs=float(np.abs(x-y).max()), changed_decisions=int(np.count_nonzero(x.argmax(1) != y.argmax(1))))


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    d, model, qf, provenance = source()
    inputs = arms(d)
    aligned, blob, frozen, trace_hash = export(d, model, qf, inputs)
    qf.close()
    template, scope = shipped()
    template.blobs['fbl'] = blob
    stats, scores = {}, {}
    for visit in DAYS:
        for arm in ARMS:
            scores[visit, arm], stats[visit, arm] = consume(template, scope, aligned[visit], inputs[visit][arm])
    for visit in reversed(DAYS):
        for arm in reversed(ARMS):
            replay, stats_repeat = consume(template, scope, aligned[visit], inputs[visit][arm])
            assert np.array_equal(replay, scores[visit, arm], equal_nan=True) and stats_repeat == stats[visit, arm]
    causal = stats['2', 'fresh_causal']
    recording = stats['2', 'fresh_recording']
    assert causal['augmentations'] == recording['augmentations']
    assert causal['optimiser'][0]['rng'] == recording['optimiser'][0]['rng']
    perturbed = aligned['2'].copy()
    perturbed[20] = -perturbed[20]
    alt, past = consume(template, scope, perturbed, inputs['2']['fresh_causal'])
    assert past == causal and not np.array_equal(alt[20], scores['2', 'fresh_causal'][20])
    ca, re = (stats['2', a]['optimiser'][0] for a in ('fresh_causal', 'fresh_recording'))
    weights = np.abs(np.asarray(ca['weights']) - re['weights'])
    new = delta(scores['2', 'fresh_recording'], scores['2', 'fresh_causal'])
    new.update(weight_max_abs=float(weights.max()), changed_weights=int(np.count_nonzero(weights)),
               changed_labels=int(np.count_nonzero(np.asarray(ca['labels']) != re['labels'])),
               parameters_differ=ca['updated_parameters'] != re['updated_parameters'])
    new['prediction_met'] = new['weight_max_abs'] > 1e-4 and new['max_abs'] > 1e-5 and new['changed_labels'] == 0
    old = delta(scores['2', 's281_archive'], scores['2', 'bridge_archive'])
    assert template.nets is None and not template.lab and template.refits == template.steps == 0
    bank = {'index|'+v: inputs[v]['s281_archive']['index'] for v in DAYS}
    bank.update({'|'.join(k): v for k, v in scores.items()})
    with Path('fbl_progress.npz').open('xb') as handle:
        np.savez_compressed(handle, **bank)
    proof = dict(status='passed_label_blind_single_family', no_gold=True, source=provenance, frozen=frozen,
                 trace_sha256=trace_hash, package_sha256=sha(PACKAGE), torch_version=str(torch.__version__),
                 input_and_source_manifest_sha256=sha(Path('fbl_inputs.sha256')), source_manifest_sha256=sha(Path('fbl_source.sha256')),
                 cases={'|'.join(k): v for k, v in stats.items()}, matched_delta=new, archive_delta=old,
                 current_window_perturbation_exact_fit=True, reverse_order_exact=True, bank_sha256=sha(Path('fbl_progress.npz')))
    with Path('fbl_progress.json').open('x') as handle:
        json.dump(proof, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(frozen=frozen, source=provenance, matched=new, archive=old), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
