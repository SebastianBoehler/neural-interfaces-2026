"""Gate each peer's packaged-style refit on its own authentic frozen parity."""
import copy
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from fbl_refit_inputs import ARMS, DAYS, PACKAGE
from peer_frozen import PEERS, definitions, difference, frozen_check, source
from returning_owner import digest, sha
from s281_seen_fbl import run_session
from s281_seen_relabel import PACKAGE_SHA


def write(name, payload):
    with Path(name).open('x') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def packaged(family, blob):
    assert sha(PACKAGE) == PACKAGE_SHA
    member = PEERS[family][3]
    with zipfile.ZipFile(PACKAGE) as archive:
        archive_source = Path('sealed_submission.py')
        with archive_source.open('xb') as f:
            f.write(archive.read('submission.py'))
        with np.load(io.BytesIO(archive.read('weights.npz')), allow_pickle=False) as z:
            weights = {k: z[k] for k in z.files}
        assert int(weights['v6_ft_member'][list(weights['v6_slots']).index(family)]) == member
        filename = str(weights[f'net{member}_sjit_file'])
        with Path(filename).open('xb') as f:
            f.write(archive.read(filename))
    scope = definitions(archive_source, {'FineTune', 'STATE', 'strong'},
                        dict(copy=copy, io=io, np=np, torch=torch, F=F, time=time, Path=Path))
    required = dict(on=1, relabel=1, labels=0, cap=250, t0=20, every=10, steps=5, seed=0, lr=.001, lam=1.)
    assert all(weights['v6_ft_'+k].item() == v for k, v in required.items())
    assert int(weights['v6_routed']) == 1
    weights.update(v6_slots=np.array([family]), v6_ft_member=np.array([member]))
    w = weights['member_weights'].astype(np.float64)
    template = scope['FineTune'](weights, w/w.sum(), 'cpu', Path('.'))
    assert template.member == {family: member}
    template.blobs[family] = blob
    return template, scope, member


def consume(family, template, scope, member, x, record):
    ft = copy.deepcopy(template)
    strong, original, step = scope['strong'], ft._refit, ft.step
    boundary = int(np.flatnonzero(record['recognised'])[0])
    actions, augmentations = [], []
    def augment(batch, rng):
        assert batch.shape == (20, 41, 480) and np.array_equal(batch.numpy(), x[:20])
        out = strong(batch, rng)
        assert not torch.equal(batch, out) and torch.isfinite(out).all()
        augmentations.append(digest(out.numpy()))
        return out
    def refit(t, lab, wt):
        assert t == 20 and boundary == 28 and len(ft.lab) == t and not actions
        p = record['prefix'][:t, 0]
        assert np.array_equal(lab, p.argmax(1)) and np.array_equal(wt, p.max(1))
        original(t, lab, wt)
        parameters = torch.cat([p.detach().flatten() for p in ft.params(ft.nets[family])]).numpy()
        assert np.isfinite(parameters).all()
        actions.append(dict(age=t, labels=lab.tolist(), weights=wt.astype(np.float32).tolist(),
                            fit=digest(parameters), rng=digest(ft.gens[family].get_state().numpy())))
    def bind_family(members, per_member, target, relabel):
        # The §281 consumer exposes FBLight's key and member ID. Keep its
        # input guards and schedule while passing this family's IDs to FineTune.
        assert set(per_member) == {59}
        out = step(members, {member: per_member[59]}, target, relabel)
        assert set(out) <= {family}
        return {'fbl': out[family]} if out else {}
    scope['strong'], ft._refit, ft.step = augment, refit, bind_family
    try:
        p, ticks = run_session(ft, x, record)
    finally:
        scope['strong'] = strong
    active = boundary == 28
    assert boundary in (10, 28) and len(ft.lab) == len(ft.wt) == boundary
    assert ticks == ([20] if active else []) and ft.refits == ft.relabels == int(active)
    assert ft.capped == ft.over == 0 and (ft.dtheta > 0) == active
    assert ft.steps == len(augmentations) == 5*int(active)
    mask = (np.arange(len(p)) >= 20) & (np.arange(len(p)) < boundary)
    assert np.array_equal(np.isfinite(p).all(1), mask) and np.isnan(p[~mask]).all()
    return p, dict(age=boundary, refits=ticks, steps=ft.steps, update=ft.dtheta, actions=actions,
                   active=int(mask.sum()), augmentations=augmentations)


def warm(family, blob, X, inputs):
    template, scope, member = packaged(family, blob)
    bank, details = {}, {}
    for visit in DAYS:
        bank['index|'+visit] = inputs[visit]['s281_archive']['index']
        for arm in ARMS:
            bank[visit+'|'+arm], details[visit+'|'+arm] = consume(family, template, scope, member, X[visit], inputs[visit][arm])
    for visit in reversed(DAYS):
        for arm in reversed(ARMS):
            p, audit = consume(family, template, scope, member, X[visit], inputs[visit][arm])
            assert np.array_equal(p, bank[visit+'|'+arm], equal_nan=True) and audit == details[visit+'|'+arm]
        assert np.array_equal(bank[visit+'|s281_archive'], bank[visit+'|bridge_archive'], equal_nan=True)
        assert details[visit+'|s281_archive'] == details[visit+'|bridge_archive']
    a, b = (details['2|'+n] for n in ('fresh_recording', 'fresh_causal'))
    aa, bb = a['actions'][0], b['actions'][0]
    assert aa['labels'] == bb['labels'] and aa['rng'] == bb['rng'] and a['augmentations'] == b['augmentations']
    weight = np.abs(np.asarray(aa['weights']) - bb['weights'])
    assert weight.max() > 1e-4 and aa['fit'] != bb['fit']
    altered = X['2'].copy()
    altered[20] = -altered[20]
    p, test = consume(family, template, scope, member, altered, inputs['2']['fresh_causal'])
    assert b == test and not np.array_equal(p[20], bank['2|fresh_causal'][20])
    metrics = difference(bank['2|fresh_recording'][20:28], bank['2|fresh_causal'][20:28])
    with Path('warm.npz').open('xb') as handle:
        np.savez_compressed(handle, **bank)
    return dict(cases=details, unchanged_archive=True, reverse_reset=True, current_excluded=True, warm=metrics,
                predicted_change_met=metrics['max_abs'] > 1e-5, weights_changed=int(np.count_nonzero(weight)),
                weight_max_delta=float(weight.max()), warm_sha256=sha(Path('warm.npz')))


def main(family, device):
    assert family in PEERS and device in ('cpu', 'cuda')
    if device == 'cuda':
        assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, inputs, X, ids, ref, source_audit = source(family)
    blob, frozen, passed, trace = frozen_check(family, device, model, inputs, X, ids, ref)
    report = dict(family=family, device=device, torch=str(torch.__version__), source=source_audit, frozen=frozen,
                  trace_sha256=trace, frozen_passed=passed, no_gold=True, source_manifest=sha(Path('../peer_source.sha256')),
                  inputs_manifest=sha(Path('../peer_inputs.sha256')))
    write('frozen.json', report)
    print(json.dumps(report, sort_keys=True), flush=True)
    if not passed:
        raise ValueError('authentic full0 frozen parity failed; warm outputs withheld')
    if device == 'cpu':
        result = warm(family, blob, X, inputs)
        write('warm.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
