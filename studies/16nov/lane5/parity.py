"""Native/Python reference parity and session isolation; never scores labels."""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(os.environ.get('NI26_WS', Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(ROOT / 'studies/16nov/lane5'), str(ROOT / 'studies/16nov/lane2')]
from model import Head, features  # noqa: E402
from bridge import Native, Session  # noqa: E402


def difference(a, b, atol, rtol, name):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'{name}: invalid shapes or values')
    diff = float(np.max(np.abs(a - b)))
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise ValueError(f'{name}: max absolute difference {diff}, atol={atol}, rtol={rtol}')
    return diff


def flat(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def session_checks(solver, windows):
    if len(windows) < 61:
        raise ValueError('session isolation needs at least 61 windows')
    a, b = Session(solver), Session(solver)
    for x in windows[:19]:
        a.step(x, True)
    branch = copy.deepcopy(a)
    for i, x in enumerate(windows[19:61]):
        y = a.step(x, True)
        b.step(windows[60 - i], True)
        other = branch.step(x, True)
        difference(y, other, 1e-6, 1e-6, 'clone and interleave')
    disabled = Session(solver)
    for x in windows[:61]:
        difference(disabled.step(x, False), solver.predict(x[None])[0], 1e-6, 1e-6, 'update disabled')
    off = disabled.solver.status()
    if off['windows'] != 61 or off['updates'] != 0 or off['delta'] != 0:
        raise ValueError(f'update-disabled session changed state: {off}')
    st = a.solver.status()
    if (st['windows'] != 61 or st['updates'] != 2 or st['prefix_count'] != 60 or
            not 0 < st['max_grad'] or not 0 < st['delta'] <= 0.100001):
        raise ValueError(f'bad session update status: {st}')
    a.reset()
    if a.solver.status()['windows'] != 0 or a.solver.status()['updates'] != 0:
        raise ValueError('reset did not clear session counters')
    fresh = Session(solver)
    for x in windows[:10]:
        difference(a.step(x, True), fresh.step(x, True), 1e-6, 1e-6, 'reset')
    first, altered = Session(solver), Session(solver)
    for i, window in enumerate(windows[:61]):
        left = first.step(window, True)
        right = altered.step(window if i < 50 else window * 1.5, True)
        if i < 50:
            difference(left, right, 1e-6, 1e-6, 'future-prefix independence')
    return st


def main():
    torch.set_num_threads(1)
    native = Native(os.environ['LANE5_LIB'])
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        subject, session = z['subject'], z['session']
        train = np.flatnonzero(np.char.endswith(subject, '/P02'))[:6]
        test = np.flatnonzero(np.char.endswith(subject, '/P01') & (session == '1'))[:61]
        if len(train) != 6 or len(test) != 61:
            raise ValueError('missing fixed parity windows')
        x, held = z['X'][train].copy(), z['X'][test].copy()
    device = int(os.environ['LANE5_DEVICE'])
    if device not in (0, 1):
        raise ValueError('LANE5_DEVICE must be 0 or 1')
    report = {'held_labels_read': False, 'native_device': device}
    reference = features(x)
    report['feature_max_abs'] = difference(native.features(x), reference, 2e-5, 2e-6, 'features')
    rng = np.random.default_rng(1341)
    mu = reference.mean(0)
    sd = np.maximum(reference.std(0), 1e-4)
    xx = np.clip((reference - mu) / sd, -8, 8).astype(np.float32)
    mask = (rng.random(xx.shape) >= .1).astype(np.float32)
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    weights = np.array([.5, 1., 1.5, .7, 1.3, 1.], dtype=np.float32)
    torch.manual_seed(47); model = Head(); model.eval()
    net = native.create(0, device, 47); net.set_norm(mu, sd); net.set_params(flat(model))
    report['parameter_set_max_abs'] = difference(net.params(), flat(model), 0, 0, 'parameters')
    v = torch.tensor(xx, requires_grad=True)
    with torch.no_grad(): expected = model(v).numpy()
    actual = net.head_forward(xx)
    report['logit_max_abs'] = difference(actual, expected, 2e-5, 2e-5, 'logits')
    report['full_raw_prediction_max_abs'] = difference(net.predict(x), expected, 2e-5, 2e-5, 'raw prediction')
    outside = features(held[:2])
    unclipped = (outside - mu) / sd
    clipped = np.clip(unclipped, -8, 8).astype(np.float32)
    report['clipped_feature_values'] = int(np.count_nonzero(np.abs(unclipped) > 8))
    if not report['clipped_feature_values']:
        raise ValueError('out-of-fit windows did not exercise the normalization clamp')
    with torch.no_grad(): clipped_expected = model(torch.from_numpy(clipped)).numpy()
    report['clipped_raw_prediction_max_abs'] = difference(
        net.predict(held[:2]), clipped_expected, 2e-5, 2e-5, 'clipped raw prediction')
    ranked = np.sort(expected, axis=1)
    non_tied = (ranked[:, -1] - ranked[:, -2]) > 2e-5
    report['nontied_real_windows'] = int(non_tied.sum())
    report['nontied_label_differences'] = int(np.sum(actual[non_tied].argmax(1) != expected[non_tied].argmax(1)))
    if not non_tied.any() or report['nontied_label_differences']:
        raise ValueError('non-tied real-window label parity failed')
    loss = (F.cross_entropy(model(v * torch.from_numpy(mask)), torch.from_numpy(labels), reduction='none') *
            torch.from_numpy(weights)).mean()
    loss.backward()
    nl, ndx, ndp = net.head_grads(xx, labels, weights, mask)
    report['loss_abs'] = difference([nl], [loss.item()], 2e-5, 2e-5, 'loss')
    report['input_grad_max_abs'] = difference(ndx, v.grad.numpy(), 2e-5, 2e-5, 'input gradient')
    pg = np.concatenate([p.grad.detach().numpy().ravel() for p in model.parameters()])
    report['parameter_grad_max_abs'] = difference(ndp, pg, 2e-5, 2e-5, 'parameter gradient')
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001); opt.step()
    net.adam_step(xx, labels, weights, mask, .001, .001)
    report['adamw_max_abs'] = difference(net.params(), flat(model), 2e-5, 2e-5, 'one AdamW step')
    # Checkpoint contains the explicit population anchor, not the transient Adam step.
    net.set_params(flat(model))
    with tempfile.TemporaryDirectory(prefix='ni26_lane5_') as tmp:
        path = Path(tmp) / 'anchor.pt'; net.save(path)
        loaded = native.create(0, device, 999); loaded.load(path)
        report['checkpoint_param_max_abs'] = difference(loaded.params(), net.params(), 0, 0, 'checkpoint')
        report['checkpoint_logit_max_abs'] = difference(loaded.predict(x), net.predict(x), 1e-6, 1e-6, 'checkpoint logits')
        report['session_status'] = session_checks(loaded, held)
    net.close(); loaded.close()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
