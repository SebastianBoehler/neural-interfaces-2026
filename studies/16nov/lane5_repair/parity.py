"""Fixed synthetic CPU parity and session-isolation checks; no treatment score."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(os.environ['NI26_WS']).resolve()
sys.path.insert(0, str(ROOT / 'studies/16nov/lane5_repair'))
from bridge import RepairNative  # noqa: E402

spec = importlib.util.spec_from_file_location('original_bridge', ROOT / 'studies/16nov/lane5/bridge.py')
original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)


def model_parts(values):
    n = 64 * 246
    return (torch.tensor(values[:n].reshape(64, 246)), torch.tensor(values[n:n + 64]),
            torch.tensor(values[n + 64:n + 64 + 192].reshape(3, 64)), torch.tensor(values[-3:]))


def expected(features, params, mode):
    w0, b0, anchor_w, anchor_b = model_parts(params)
    w, b = anchor_w.clone(), anchor_b.clone()
    logits, counts = [], []
    for t, x in enumerate(features):
        if t >= 40 and t % 20 == 0:
            prefix = torch.tensor(features[max(0, t - 64):t])
            hidden = F.gelu(F.linear(prefix, w0, b0), approximate='none')
            qlog = F.log_softmax(F.linear(hidden, anchor_w, anchor_b), 1)
            q = qlog.exp(); chosen = q.max(1).values >= .70
            count = int(chosen.sum()); counts.append(count)
            if count >= 8:
                h, q, qlog = hidden[chosen], q[chosen], qlog[chosen]
                for _ in range(3):
                    w.requires_grad_(True); b.requires_grad_(True)
                    logp = F.log_softmax(F.linear(h, w, b), 1); p = logp.exp()
                    entropy = -(p * logp).sum(1).mean()
                    mean = p.mean(0).clamp_min(1e-12)
                    diversity = -(mean * mean.log()).sum()
                    kl = (q * (qlog - logp)).sum(1).mean()
                    penalty = (w - anchor_w).square().sum() + (b - anchor_b).square().sum()
                    loss = (entropy if mode == 1 else -entropy) - .1 * diversity + kl + .3 * penalty
                    gw, gb = torch.autograd.grad(loss, (w, b))
                    with torch.no_grad():
                        w = (w - .001 * gw).detach(); b = (b - .001 * gb).detach()
                        delta = ((w - anchor_w).square().sum() + (b - anchor_b).square().sum()).sqrt()
                        scale = min(1., .03 / max(float(delta), 1e-12))
                        w = anchor_w + scale * (w - anchor_w); b = anchor_b + scale * (b - anchor_b)
        with torch.no_grad():
            h = F.gelu(F.linear(torch.tensor(x), w0, b0), approximate='none')
            logits.append(F.linear(h, w, b).numpy())
    return np.stack(logits), counts


def run():
    torch.set_num_threads(1)
    native = original.Native(os.environ['LANE5_LIB']); extension = RepairNative(os.environ['LANE5_REPAIR_LIB'])
    population = native.create(0, 0, 47)
    params = population.params(); params[-3:] = (4., -1., -1.); population.set_params(params)
    windows = np.random.default_rng(921).normal(size=(61, 41, 480)).astype(np.float32)
    features = np.clip(native.features(windows), -8, 8)
    initial = population.params().copy(); before = population.predict(windows[:1])[0]
    disabled = original.Session(population)
    for x in windows:
        if not np.array_equal(disabled.step(x, False), population.predict(x[None])[0]):
            raise ValueError('legacy disabled path changed predictions')
    if disabled.solver.status()['updates']:
        raise ValueError('legacy disabled path updated')
    disabled.solver.close()
    dirty = population.clone(); dirty.step(windows[0], False)
    try:
        extension.create(dirty, 1)
    except RuntimeError:
        pass
    else:
        raise ValueError('repair accepted a non-fresh population')
    dirty.close()
    reports = {}
    for mode in (1, 2):
        a, b = extension.create(population, mode), extension.create(population, mode)
        before_teacher = a.teacher(windows[0]); observed, interleaved = [], []
        for x in windows:
            observed.append(a.step(x)); interleaved.append(b.step(x))
        if not np.array_equal(observed, interleaved):
            raise ValueError("interleaved sessions changed each other")
        expected_logits, counts = expected(features, params, mode)
        diff = float(np.max(np.abs(np.asarray(observed) - expected_logits)))
        if not np.allclose(observed, expected_logits, rtol=2e-5, atol=2e-5):
            raise ValueError(f'Python policy parity failed mode={mode} diff={diff}')
        status = a.status()
        if status['selections'] != counts or status['updates'] != 2 or status['skips'] or status['delta'] > .030001 or status['max_grad'] <= 0:
            raise ValueError(f'inactive or invalid updates: {status}')
        if not np.array_equal(before_teacher, a.teacher(windows[0])) or not np.array_equal(initial, population.params()):
            raise ValueError('teacher or population changed')
        a.reset()
        if a.status()['windows'] or not np.array_equal(a.step(windows[0]), b.teacher(windows[0])):
            raise ValueError('session reset did not restore the population')
        if not np.array_equal(b.teacher(windows[0]), before):
            raise ValueError('interleaved session changed frozen teacher')
        reports[str(mode)] = {'max_abs': diff, 'status': status}
        a.close(); b.close()
    # The same prefix must not depend on different future suffixes.
    left, right = extension.create(population, 1), extension.create(population, 1)
    left_prefix, right_prefix = [], []
    for x in windows:
        left_prefix.append(left.step(x)); right_prefix.append(right.step(x))
    left.step(windows[0]); right.step(-windows[0])
    if not np.array_equal(left_prefix, right_prefix):
        raise ValueError('prefix changed with different future windows')
    left.close(); right.close()
    zero = native.create(0, 0, 3); zero.set_params(np.zeros_like(zero.params()))
    inactive = extension.create(zero, 1)
    for x in windows:
        inactive.step(x)
    no_op = inactive.status()
    if no_op['attempts'] != 2 or no_op['updates'] or no_op['skips'] != 2 or no_op['selected_total']:
        raise ValueError(f'no-confidence skip incorrect: {no_op}')
    inactive.close(); zero.close(); population.close()
    print('CPU POLICY PARITY OK', reports, 'zero_selected', no_op, flush=True)


if __name__ == '__main__':
    run()
