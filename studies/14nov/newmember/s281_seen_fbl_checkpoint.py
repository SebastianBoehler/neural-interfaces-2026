"""Compare saved FBLight seen-fold checkpoints to archived frozen QF outputs.

No fine-tuning or accuracy score. Failure means these checkpoints cannot be
used as the registered QF baseline without an independently validated rebuild.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'scripts')]
import deep
import protocol
import session_bn as SB

CHECKPOINT_SHA = {'P04_s23': 'c45b30f8650204346c23ac8e73a4a9534ce00d09157646d0b91f716ca2cd5ad2',
                  'P10_s23': '1d5460ed0977199251026696912758491c3b5207993244a44539da270c20d47e'}
ARCHIVE_SHA = '220649ed1f6b86836d09860965a5f0115d5faeee4f8e0e3784255ddcaf8dfb96'
CHECKPOINTS = {0: CHECKPOINT_SHA,
               1: {'P04_s23': '7cd65701c057196b807e6748baca94f78016c77715e6016f308c4ea98347157b',
                   'P10_s23': 'a7f6315fe2b61d4191f083231f24ebb99b644311eea3ea7335b56033458daa78'}}
ARCHIVES = {0: ARCHIVE_SHA, 1: '5bb4a79845d5067eb471a27df9e567477ac9d0410223881a74215e2790de0837'}


def load_fold(d, fold, seed=0):
    assert seed in CHECKPOINTS and fold in CHECKPOINTS[seed], 'unsupported fold or seed'
    masks = {name: (tr, te) for name, tr, te in protocol.folds(d, 'seen_session')}
    train, held = masks[fold]; who = fold.split('_')[0]
    own = d['stream'] == f'Dreyer2026Proteus/{who}|1'
    assert not (train & held).any() and (own & train).sum() == own.sum() > 0 and not (own & held).any()
    assert set(d['session'][held]) == {'2', '3'} and set(d['subject'][held]) == {f'Dreyer2026Proteus/{who}'}
    path = ROOT / f'runs/diag/pdeep/nets/fbl-{fold}-s{seed}.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CHECKPOINTS[seed][fold], 'checkpoint hash mismatch'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert set(checkpoint) == {'state', 'G', 'scale'}
    m = SB.make('fblight_ea_rot05_sbn', d, seed); m.dev = 'cpu'
    m.G, m.scale = checkpoint['G'], checkpoint['scale']
    dims = m._aligned(np.flatnonzero(train)[:2], False).shape[1:]
    m.net = deep.build(m.arch, *dims, m.pretrained, **m.net_kw).to('cpu')
    if m.train_fir:
        for p in m.net.spectral_filtering.parameters(): p.requires_grad_(True)
    m._prepare(); m.net.load_state_dict(checkpoint['state'], strict=True); m.net.eval(); m.set_k(5.)
    return m, held


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    seed = int(sys.argv[1]) if len(sys.argv) == 2 else 0
    assert seed in CHECKPOINTS, 'unsupported seed'
    d = protocol.load(); archive_path = ROOT / f'runs/ttt/probs18/ttt18within0QF_fbl_base-seen_session-s{seed}.npz'
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == ARCHIVES[seed], 'archive hash mismatch'
    with np.load(archive_path) as q:
        indices, expected = q['index'].copy(), q['proba'].copy()
    with np.load(ROOT / 'runs/shuffle_perms.npz') as p:
        delivery = p['within|0|seen_session'].copy()
    assert len(indices) == 762 and np.array_equal(np.sort(delivery), indices)
    fold_masks = list(protocol.folds(d, 'seen_session')); found = []
    for fold, train, held in fold_masks:
        ckpt_path = ROOT / f'runs/diag/pdeep/nets/fbl-{fold}-s{seed}.pt'
        m, checked_held = load_fold(d, fold, seed); assert np.array_equal(checked_held, held)
        rows_all = np.flatnonzero(held); outputs = []
        for session in dict.fromkeys(d['stream'][rows_all]):
            rows = delivery[d['stream'][delivery] == session]; assert len(rows) == (d['stream'][rows_all] == session).sum()
            X = torch.from_numpy(np.ascontiguousarray(m._aligned(rows, causal=True), dtype=np.float32))
            with torch.no_grad(): pred = torch.softmax(m.net(X)[:, :3].double(), 1).numpy()
            ref_pos = np.searchsorted(indices, rows); assert np.array_equal(indices[ref_pos], rows)
            assert np.isfinite(pred).all() and np.isfinite(expected[ref_pos]).all()
            err = np.abs(pred - expected[ref_pos]); labels = int((pred.argmax(1) != expected[ref_pos].argmax(1)).sum())
            outputs.append((float(err.max()), labels)); found.extend(rows.tolist())
            print(f'{fold} {session} n={len(rows)} max_abs={err.max():.9g} label_differences={labels}', flush=True)
        print(f'{fold} checkpoint_sha256={hashlib.sha256(ckpt_path.read_bytes()).hexdigest()} max_abs={max(x[0] for x in outputs):.9g}', flush=True)
        assert max(x[0] for x in outputs) <= 1e-5 and sum(x[1] for x in outputs) == 0, 'frozen archive parity failed'
    assert np.array_equal(np.sort(found), indices), 'held-out rows not covered exactly once'
    print('archive_sha256=' + hashlib.sha256(archive_path.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    assert len(sys.argv) <= 2 and (len(sys.argv) == 1 or sys.argv[1] in ('0', '1')), 'seed must be 0 or 1'
    main()
