"""Validated seed-0 held-out FBLight FineTune adapter; no scoring or output files.

Only within0, B2_s0, B2_s1 and B2_pool are currently enabled. Each session requires
matching base-specific prefix/route archives and frozen QF equality. The CLI
provides a single non-scoring implementation smoke on P10 session 2.
"""
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT / 'experiments'), str(ROOT / 'scripts'), str(ROOT)]
import protocol
import session_bn as SB
from export_external_teacher import device_free_trace
from s281_seen_fbl import session_inputs, shipped_ft, run_session
from s281_seen_fbl_checkpoint import ARCHIVES, load_fold

_TOKEN = object()


def seal(record, inputs, trace_bytes):
    h = hashlib.sha256(trace_bytes)
    for value in (inputs, *(record[k] for k in ('index', 'prefix', 'grid', 'emitted', 'recognised'))):
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode()); h.update(a.tobytes())
    return h.hexdigest()


class VerifiedSession:
    """Only construct with prepare: fold, order, preprocessing and QF are validated."""
    def __init__(self, token, record, inputs, trace_bytes, checks):
        assert token is _TOKEN, 'use prepare to verify the session'
        self.record, self.inputs, self.trace_bytes, self.checks = record, inputs, trace_bytes, checks
        self.seal = seal(record, inputs, trace_bytes)


def prepare(order, base, session, d):
    seeds = {'B2_s0': 0, 'B2_pool': 0, 'B2_s1': 1}
    assert order == 'within0' and base in seeds, 'no verified checkpoint/prefix for this order or base'
    assert session in ('Dreyer2026Proteus/P04|2', 'Dreyer2026Proteus/P04|3',
                       'Dreyer2026Proteus/P10|2', 'Dreyer2026Proteus/P10|3'), 'unsupported session'
    fold = session.split('/')[1].split('|')[0] + '_s23'
    seed = seeds[base]
    record = session_inputs(order, base, session, d); model, held = load_fold(d, fold, seed)
    rows = record['index']; assert held[rows].all(), 'session is not held out'
    X = np.ascontiguousarray(model._aligned(rows, causal=True), dtype=np.float32)
    assert X.shape == (len(rows), 41, 480) and np.isfinite(X).all(), 'invalid aligned inputs'
    archive = ROOT / f'runs/ttt/probs18/ttt18within0QF_fbl_base-seen_session-s{seed}.npz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVES[seed], 'QF archive hash mismatch'
    with np.load(archive) as z:
        idx, expected = z['index'], z['proba']; pos = np.searchsorted(idx, rows)
        assert np.all(pos < len(idx)) and np.array_equal(idx[pos], rows), 'QF rows mismatch'
        expected = expected[pos].copy()
    stream_net = SB.to_stream(model.net, 5.0).cpu().eval()
    trace = device_free_trace(stream_net, torch.from_numpy(X[:4])); SB.reset_stream(trace)
    buffer = io.BytesIO(); torch.jit.save(trace, buffer); blob = buffer.getvalue()
    traced = torch.jit.load(io.BytesIO(blob), map_location='cpu').eval(); SB.reset_stream(traced)
    with torch.no_grad():
        tensor = torch.from_numpy(X)
        eager = torch.softmax(model.net(tensor)[:, :3].double(), 1).numpy()
        actual = torch.softmax(traced(tensor)[:, :3].double(), 1).numpy()
    assert np.isfinite(eager).all() and np.isfinite(actual).all() and np.isfinite(expected).all()
    de, dt = float(np.max(np.abs(eager - expected))), float(np.max(np.abs(actual - eager)))
    changed = int(np.sum(eager.argmax(1) != expected.argmax(1)))
    assert de <= 1e-5 and dt <= 1e-5 and changed == 0, f'frozen model mismatch: {de}, {dt}, {changed}'
    return VerifiedSession(_TOKEN, record, X, blob, dict(fold=fold, base=base, seed=seed, archive_max_diff=de, trace_max_diff=dt,
                                               changed_labels=changed, trace_sha256=hashlib.sha256(blob).hexdigest()))


def consume(verified):
    assert isinstance(verified, VerifiedSession), 'verified session required'
    assert verified.seal == seal(verified.record, verified.inputs, verified.trace_bytes), 'verified inputs changed'
    ft, _ = shipped_ft()
    ft.blobs['fbl'] = verified.trace_bytes  # only after fold/trace/stream/archive validation
    pred, refits = run_session(ft, verified.inputs, verified.record)
    return pred, refits, dict(refits=ft.refits, relabels=ft.relabels, max_update=ft.dtheta)


def main():
    assert sys.argv[1:] in (['smoke'], ['smoke-s1']), 'only non-scoring P10 implementation smokes are enabled'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    d = protocol.load(); base = 'B2_s1' if sys.argv[1] == 'smoke-s1' else 'B2_s0'
    s = prepare('within0', base, 'Dreyer2026Proteus/P10|2', d)
    pred, refits, metrics = consume(s)
    assert refits == [20, 30, 40, 50] and metrics['refits'] == metrics['relabels'] == 4 and metrics['max_update'] > 0
    assert np.isnan(pred[:20]).all() and np.isfinite(pred[20:60]).all() and np.isnan(pred[60:]).all()
    print(f'fold-specific frozen checks={s.checks}; refits={refits}; metrics={metrics}', flush=True)
    print('S281 SEEN FBL DEV SMOKE PASS (no score, no prediction artifact)', flush=True)


if __name__ == '__main__': main()
