"""Capture seen-session v5 gate scores, ramps and identities without scoring.

This preserves the original gate. The output is a dev-gate emulation, not a
worker/package-gate identity claim. Existing trace files are not modified.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
from s281_seen_route import BASES, ORDERS, latch_traces, load_routing


def pack(traces, rows):
    lookup = {int(g): i for i, g in enumerate(rows)}
    assert len(lookup) == len(rows) and np.all(np.diff(rows) > 0), 'output rows must be unique and sorted'
    fields = ('contrast_r', 'contrast_j', 'contrast_score', 'bio_r', 'bio_j', 'bio_score',
              'winner_r', 'winner_j', 'contrast_person', 'bio_person', 'winner_person')
    out = {k: np.full(len(rows), '', dtype='<U64') if k.endswith('_person') else
           np.full(len(rows), -1, dtype=np.int64) if k.endswith('_j') else
           np.full(len(rows), np.nan, dtype=np.float64) for k in fields}
    seen = set()
    for t in traces:
        idx = np.asarray(t['rows'], int); assert all(len(str(n)) <= 64 for n in t['names'])
        names = np.asarray(t['names'], dtype='<U64')
        assert len(idx) and len(names) and len(set(names.tolist())) == len(names)
        assert not seen.intersection(idx.tolist()) and all(int(g) in lookup for g in idx)
        seen.update(idx.tolist()); target = [lookup[int(g)] for g in idx]
        cr, cj, cs, br, bj, bs = (np.asarray(t[k]) for k in ('contrast_r', 'contrast_j', 'contrast_score', 'bio_r', 'bio_j', 'bio_score'))
        assert all(x.shape == idx.shape for x in (cr, cj, cs, br, bj, bs))
        for ramp, who, score in ((cr, cj, cs), (br, bj, bs)):
            assert np.isfinite(ramp).all() and ((ramp >= 0) & (ramp <= 1)).all()
            assert np.issubdtype(who.dtype, np.integer) and ((who >= -1) & (who < len(names))).all()
            assert np.isfinite(score[who >= 0]).all() and not (ramp[who < 0] > 0).any()
        use_bio = br > cr; wr, wj = np.where(use_bio, br, cr), np.where(use_bio, bj, cj)
        def people(who):
            result = np.full(len(who), '', dtype='<U64'); ok = who >= 0; result[ok] = names[who[ok]]; return result
        for k, v in dict(contrast_r=cr, contrast_j=cj, contrast_score=cs, bio_r=br, bio_j=bj, bio_score=bs,
                         winner_r=wr, winner_j=wj, contrast_person=people(cj), bio_person=people(bj), winner_person=people(wj)).items():
            out[k][target] = v
    assert seen == set(lookup), 'trace does not cover exactly the output rows'
    return out


def generate(order):
    assert order in ORDERS
    module = load_routing(); K, S, A, ns, proto = (module[k] for k in ('K', 'S', 'A', 'ns', 'p'))
    rows = module['IDX']; p = module['perm'](order); pp = p[proto]
    reference_path = ROOT / f'runs/s281_seen_route/{order}.npz'
    with np.load(reference_path) as z: reference = {k: z[k] for k in z.files}
    assert np.array_equal(reference['index'], rows) and str(reference['order']) == order
    oof = {b: module['oof_rec'](b) for b in BASES}
    for cache in (ns['_BIO'], A._H, K.STREAMS, K.REC): cache.clear()
    S.apply(p); K.RANK['cur'] = {q: K._rank(K.IDX[q][p[q]]) for q in K.PR}
    module['apply_members'](module['member_deltas'](order), pp)
    orig_rmap, orig_bio, old_folds = A.R3.rmap, ns['bio_stream'], A.RR.FOLDS
    pending, traces = [], []
    def rmap(scores, *args):
        r, j = orig_rmap(scores, *args); scores = np.asarray(scores); best = np.full(len(j), np.nan)
        ok = j >= 0; best[ok] = scores[np.flatnonzero(ok), j[ok]]
        pending.append((r.copy(), j.astype(int).copy(), best)); return r, j
    def bio(p0, fold, session, idx, names, t0):
        result = orig_bio(p0, fold, session, idx, names, t0)
        assert p0 == proto and pending, 'unmatched gate call'
        cr, cj, cs = pending.pop(0); br, bj, bs = result
        traces.append(dict(name=session, rows=np.asarray(idx, int), names=list(names), contrast_r=cr, contrast_j=cj,
                           contrast_score=cs, bio_r=np.array(br), bio_j=np.array(bj, dtype=int), bio_score=np.array(bs)))
        return result
    A.R3.rmap, ns['bio_stream'] = rmap, bio
    result = {'index': rows, 'order': np.array(order),
              'reference_sha256': np.array(hashlib.sha256(reference_path.read_bytes()).hexdigest())}
    try:
        A.RR.FOLDS = {proto: old_folds[proto]}
        for base in BASES:
            traces.clear(); pending.clear(); K.REC.clear()
            mix, a = A.mix(base, proto, list(A.FAMS)); _, shadow = A.shadow(mix, proto)
            out0 = ns['route4']({proto: mix}, {proto: shadow}, a, oof[base],
                                {proto: A.members(base, proto)['c1']}, 'max', 10)[0]
            assert not pending and len(traces) == 4
            encoded = pack(traces, rows)
            latches = latch_traces([dict(rows=t['rows'], contrast=t['contrast_r'], bio=t['bio_r']) for t in traces], rows)
            assert np.array_equal(latches, reference[f'{base}|recognised']), 'latched trace mismatch'
            for t in traces:
                key = (proto, t['name']); merged = encoded['winner_r'][[np.searchsorted(rows, x) for x in t['rows']]]
                if (merged > 0).any():
                    assert key in K.REC and np.array_equal(merged, K.REC[key][0]), 'winning ramp mismatch'
                    ix = [np.searchsorted(rows, x) for x in t['rows']]
                    assert np.array_equal(encoded['winner_j'][ix], K.REC[key][1]), 'winning identity mismatch'
                    assert list(K.NM[key][0]) == t['names'], 'gallery order mismatch'
                else:
                    assert key not in K.REC
            routed = K.staged(out0)[0][proto][np.argsort(pp)]
            assert np.allclose(routed, reference[f'{base}|routed_v51b'], atol=1e-9, rtol=0), 'replay output mismatch'
            for key, value in encoded.items(): result[f'{base}|{key}'] = value
            print(f'{order} {base}: windows={len(rows)} routed_windows={int((encoded["winner_r"] > 0).sum())} verified', flush=True)
    finally:
        A.RR.FOLDS = old_folds; A.R3.rmap, ns['bio_stream'] = orig_rmap, orig_bio
    return result


def main():
    assert len(sys.argv) == 2 and sys.argv[1] in ORDERS, 'supply a registered order'
    order = sys.argv[1]; path = ROOT / f'runs/s281_seen_gate_trace/{order}.npz'
    assert not path.exists(), f'refusing to overwrite {path}'
    data = generate(order); path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **data)
    print(f'saved {path}', flush=True)


if __name__ == '__main__': main()
