"""Fit each source-person fold once, then stream all eight orders independently."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / 'experiments')]
import protocol
import session_bn as SB
from patch import install

ARMS = ('plain', 'unconditional10', 'conditional10', 'conditional30')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(net):
    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make(d, arm, seed, epochs=None):
    if arm not in ARMS:
        raise ValueError('unregistered CORAL repair arm')
    model = SB.make('fb_sbn_rot05', d, seed)
    if not str(model.dev).startswith('cuda'):
        raise RuntimeError('CORAL training must run on a Galvani GPU')
    model.coral = 0. if arm == 'plain' else (30. if arm == 'conditional30' else 10.)
    model.repair_mode = 'conditional' if arm.startswith('conditional') else 'unconditional'
    model.repair_stats = dict(batches=0, zero_pair_batches=0, class_pairs=[0, 0, 0], gradient_norm=0.)
    if epochs is not None:
        model.epochs = epochs
    return model


def stream(model, d, rows):
    """The original causal EA + vectorized SessionBN, with no mutable stream state."""
    model.net.eval()
    with torch.no_grad():
        x = torch.from_numpy(model._aligned(rows, causal=True)).to(model.dev)
        result = torch.softmax(model.net(x)[:, :3].double(), 1).cpu().numpy()
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1, atol=1e-12):
        raise ValueError('invalid streamed probabilities')
    return result


def infer(model, d, test):
    before = weights(model.net); result = {}
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        for order in ORDERS:
            permutation = z[f'{order[:-1]}|{order[-1]}|all_loso'].astype(np.int64)
            if not np.array_equal(np.sort(permutation), np.arange(len(test))):
                raise ValueError('invalid delivery permutation')
            rows = permutation[test[permutation]]; output = np.empty((len(rows), 3))
            for session in dict.fromkeys(d['stream'][rows]):
                loc = np.flatnonzero(d['stream'][rows] == session)
                session_rows = rows[loc]
                output[loc] = stream(model, d, session_rows)
                # A second call must reset completely; adding future rows cannot alter a prefix.
                prefix = session_rows[:min(61, len(session_rows))]
                again = stream(model, d, prefix)
                if not np.allclose(again, output[loc][:len(prefix)], atol=1e-6, rtol=1e-6):
                    raise ValueError('causal prefix/session reset mismatch')
            if weights(model.net) != before:
                raise ValueError('base inference changed fitted parameters or buffers')
            result[f'{order}|index'], result[f'{order}|proba'] = rows, output
    return result, before


def main(arm, seed, fold_index):
    if Path.cwd().resolve() != ROOT or not 0 <= seed < 6 or not 0 <= fold_index < 10:
        raise ValueError('invalid working directory, seed or held-person fold')
    manifest = ROOT / 'coral304.sha256'
    if not manifest.is_file():
        raise ValueError('missing immutable source/input manifest')
    d = protocol.load(); held, train, test = list(protocol.folds(d, 'all_loso'))[fold_index]
    fit_people = sorted(set(d['subject'][train])); held_people = sorted(set(d['subject'][test]))
    if len(fit_people) != 9 or len(held_people) != 1 or set(fit_people) & set(held_people):
        raise ValueError('invalid leave-one-person-out training rows')
    output = ROOT / f'runs/coral304/{arm}_s{seed}_{held}.npz'
    if output.exists():
        raise FileExistsError(output)
    install(SB)
    model = make(d, arm, seed)
    started = time.monotonic(); model.fit(np.flatnonzero(train)); fitted = time.monotonic()
    if set(model.fit_subjects) != set(fit_people):
        raise ValueError('actual training people do not match the fold')
    if model.coral and (model.dg_stats['coral_pairs'] <= 0 or model.dg_stats['coral_pen'] <= 0):
        raise ValueError('inactive covariance penalty')
    if arm.startswith('conditional') and model.repair_stats['gradient_norm'] <= 0:
        raise ValueError('conditional penalty did not change a gradient')
    model.set_k(2.)
    result, parameter_sha = infer(model, d, test)
    metadata = dict(arm=arm, seed=seed, held=held, fit_people=fit_people,
                    train_rows_sha256=hashlib.sha256(np.flatnonzero(train).astype(np.int64).tobytes()).hexdigest(),
                    manifest_sha256=sha(manifest), parameter_sha256=parameter_sha,
                    data_sha256=sha(ROOT / 'data/proteus_windows.npz'),
                    permutation_sha256=sha(ROOT / 'runs/shuffle_perms.npz'),
                    penalty=model.repair_stats, dg_stats=model.dg_stats,
                    training_seconds=fitted-started, total_seconds=time.monotonic()-started,
                    torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0),
                    reference='original fb_sbn_rot05; conditional penalty only; fit once per fold/seed')
    output.parent.mkdir(parents=True, exist_ok=True)
    result['metadata'] = json.dumps(metadata, sort_keys=True)
    with output.open('xb') as f:
        np.savez_compressed(f, **result)
    print(f'saved {output}; no accuracy scored', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
