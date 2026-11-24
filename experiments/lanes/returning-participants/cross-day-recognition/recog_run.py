"""Lane Q test 1: cross-day open-set recognition of the spectral identity gate, stock vs cohort normalisation. See evidence/wave_20nov/laneQ_recog_hand.md.
python laneQ/recog_run.py PKGDIR FEATDIR OUT.json"""
import json, sys
import numpy as np
sys.path.insert(0, "laneQ")
from recog_lib import Gate

pk, fd, outp = sys.argv[1:4]
gate = Gate(dict(np.load(pk + "/weights.npz", allow_pickle=False)))
NORD, NSPLIT = 4, 40
CORP = {"netbci": ("q_netbci_feat.npz", 150), "ds7g": ("q_ds7g_feat.npz", 150), "ds7lr": ("q_ds7lr_feat.npz", 80)}
D = {}
for c, (fn, T) in CORP.items():
    with np.load(f"{fd}/{fn}", allow_pickle=True) as z:
        sub, ses = z["subject"].astype(str), z["session"].astype(str)
        F = gate.proj(z["F"])
    D[c] = dict(F=F, sub=sub, ses=ses, T=T, keys=sorted(set(zip(sub, ses))))
    D[c]["idx"] = {k: np.flatnonzero((sub == k[0]) & (ses == k[1])) for k in D[c]["keys"]}
    D[c]["keys"] = [k for k in D[c]["keys"] if len(D[c]["idx"][k]) >= T]  # sessions shorter than T are dropped (never padded)


def prefix_means(c, T, order, seed0=0, keys=None):
    """(sessions, T, d) running means of each session streamed in a random full shuffle (order index)."""
    d = D[c]; out = []
    for i, k in enumerate(keys or d["keys"]):
        ix = d["idx"][k]
        if len(ix) < T:
            continue  # short sessions (ds007221lr: 80 windows) are skipped, never padded
        p = np.random.default_rng([seed0, order, i]).permutation(len(ix))[:T]
        out.append(np.cumsum(d["F"][ix[p]], 0) / np.arange(1, T + 1)[:, None])
    return np.stack(out)


def session_model(c, k):
    return D[c]["F"][D[c]["idx"][k]].mean(0)


def run_corpus(c, enrol_ses, cohort_c):
    d, T = D[c], D[c]["T"]
    people = sorted(p for p in set(d["sub"]) if (p, enrol_ses) in d["idx"] and (p, enrol_ses) in set(d["keys"]))
    # candidate gallery: public 10 + every person's enrolment-session mean (m=1); columns 0..9 public, 10.. external
    G = np.vstack([gate.G0] + [session_model(c, (p, enrol_ses)) for p in people])
    m = np.r_[gate.m0, np.ones(len(people), int)]
    # cohort: models = all sessions of the other corpora; probes (Z-norm) = their sessions with >= T windows, one fixed shuffle
    cm = np.vstack([np.stack([session_model(cc, k) for k in D[cc]["keys"]]) for cc in cohort_c])
    cp = np.concatenate([prefix_means(cc, T, 99, 7) for cc in cohort_c if any(len(v) >= T for v in D[cc]["idx"].values())])
    skeys = [k for k in d["keys"] if k[0] in people]
    sess = list(enumerate(skeys))
    PM = np.stack([prefix_means(c, T, o, 0, skeys) for o in range(NORD)], 1)  # (S, orders, T, d)
    S = len(sess)
    raw = np.zeros((S, NORD, T, len(G)))
    tmu, tsd, tmuA, tsdA = (np.zeros((S, NORD, T)) for _ in range(4))
    zmu, zsd, zmuA, zsdA = (np.zeros((T, len(G))) for _ in range(4))
    top = 30
    for n in range(1, T + 1):
        x = PM[:, :, n - 1].reshape(-1, gate.d)
        raw[:, :, n - 1] = gate.score(G, m, x, n).reshape(S, NORD, -1)
        ct = np.sort(gate.score(cm, np.ones(len(cm), int), x, n), 1)[:, ::-1]  # probe vs cohort models, descending
        tmu[:, :, n - 1], tsd[:, :, n - 1] = ct.mean(1).reshape(S, NORD), ct.std(1).reshape(S, NORD)
        tmuA[:, :, n - 1], tsdA[:, :, n - 1] = ct[:, :top].mean(1).reshape(S, NORD), ct[:, :top].std(1).reshape(S, NORD)
        cz = np.sort(gate.score(G, m, cp[:, n - 1], n), 0)[::-1]  # each gallery model vs cohort probes, descending
        zmu[n - 1], zsd[n - 1], zmuA[n - 1], zsdA[n - 1] = cz.mean(0), cz.std(0), cz[:top].mean(0), cz[:top].std(0)
    sn = 0.5 * ((raw - zmu[None, None]) / zsd[None, None] + (raw - tmu[..., None]) / tsd[..., None])
    asn = 0.5 * ((raw - zmuA[None, None]) / zsdA[None, None] + (raw - tmuA[..., None]) / tsdA[..., None])
    return dict(people=people, sess=[k for _, k in sess], raw=raw, sn=sn, asn=asn, npub=len(gate.G0), T=T)


def first_latch(sc, cols, thr, persist=1, t0=11):
    """sc (R,T,M) scores; columns allowed; returns (n, j) per row for each thr in thr (R,len(thr)); n=0: never latched."""
    s = sc[:, :, cols]
    mx, j = s.max(2), np.array(cols)[s.argmax(2)]
    R, T = mx.shape
    n_out = np.zeros((R, len(thr)), int); j_out = np.full((R, len(thr)), -1)
    for a, th in enumerate(thr):
        ok = mx > th
        if persist > 1:
            run = ok.copy()
            for q in range(1, persist):
                run[:, q:] &= ok[:, :-q] & (j[:, q:] == j[:, :-q]); run[:, :q] = False
            ok = run
        ok[:, :t0 - 1] = False
        hit = ok.any(1); f = ok.argmax(1)
        n_out[hit, a] = f[hit] + 1; j_out[hit, a] = j[np.arange(R), f][hit]
    return n_out, j_out


def evaluate(res, enrol_ses, mode):
    people, sess, T, npub = res["people"], res["sess"], res["T"], res["npub"]
    pid = {p: i for i, p in enumerate(people)}
    sp = np.array([pid[k[0]] for k in sess]); isenr = np.array([k[1] == enrol_ses for k in sess])
    arms = {"stock": ("raw", [.26], 1)}
    grid = {v: np.quantile(res[v][..., npub:].max(-1)[:, :, 10:], np.linspace(0.0, .9995, 300)) for v in ("raw", "sn", "asn")}
    rows = {}
    pairs = []
    for r in range(NSPLIT // 2):
        perm = np.random.default_rng(1000 + r).permutation(len(people)); h = len(people) // 2
        pairs += [(perm[:h], perm[h:]), (perm[h:], perm[:h])]  # E, U ; swapped split calibrates the other

    def split_stats(E, U, variant, thr, persist):
        cols = (list(range(npub)) if mode == 'pub+ext' else []) + [npub + e for e in E]
        trues = np.flatnonzero(np.isin(sp, E) & ~isenr); imps = np.flatnonzero(np.isin(sp, U))
        sc = res[variant]
        out = {}
        for tag, ix in (("t", trues), ("i", imps)):
            n, j = first_latch(sc[ix].reshape(len(ix) * NORD, T, -1)[:, :, :], cols, thr, persist)
            out[tag] = (n.reshape(len(ix), NORD, -1), j.reshape(len(ix), NORD, -1), ix)
        return out

    def summarize(st, a):
        nt, jt, ixt = (x[..., a] if x.ndim == 3 else x for x in st["t"])
        ni, ji, _ = (x[..., a] if x.ndim == 3 else x for x in st["i"])
        want = npub + sp[ixt][:, None]
        ta = ((nt > 0) & (jt == want)).mean(); mis = ((nt > 0) & (jt != want)).mean(); fa = (ni > 0).mean()
        lat = nt[(nt > 0) & (jt == want)]
        return ta, mis, fa, (np.median(lat) if lat.size else np.nan), (np.mean(nt[(nt > 0) & (jt == want)] <= 30) if lat.size else 0) * ta

    stats_cache = {}
    def get(variant, persist, thr):
        k = (variant, persist, len(thr))
        if k not in stats_cache:
            stats_cache[k] = {pi: split_stats(E, U, variant, thr, persist) for pi, (E, U) in enumerate(pairs)}
        return stats_cache[k]

    def wrongrate(st, a):
        ta, mis, fa, _, _ = summarize(st, a); ntr, nim = len(st["t"][2]), len(st["i"][2])
        return (mis * ntr + fa * nim) / (ntr + nim)

    st0 = get("raw", 1, [.26])
    STOCK_W = {pi: wrongrate(st0[pi], 0) for pi in st0}
    rows["stock"] = {k: np.array([summarize(st0[pi], 0)[i] for pi in st0]) for i, k in enumerate(("TA", "misID", "FA", "med_n", "TA_by30"))}
    rows["stock"]["wrong"] = np.array([STOCK_W[pi] for pi in st0])
    for variant, persist in (("raw", 1), ("sn", 1), ("asn", 1), ("sn", 3)):
        thr = grid[variant]
        sts = get(variant, persist, thr)
        for target in ("stock", .02, .05, .10):
            acc = []
            for pi in sts:
                cal = pi ^ 1
                tw = STOCK_W[cal] if target == "stock" else target
                w = np.array([wrongrate(sts[cal], a) for a in range(len(thr))])
                ok = np.flatnonzero(w <= tw + 1e-12)
                a = ok[0] if ok.size else len(thr) - 1
                acc.append(summarize(sts[pi], a) + (wrongrate(sts[pi], a),))
            acc = np.array(acc)
            nm = f"{variant}{'_p3' if persist > 1 else ''}@{'stock' if target == 'stock' else int(target * 100)}"
            rows[nm] = dict(TA=acc[:, 0], misID=acc[:, 1], FA=acc[:, 2], med_n=acc[:, 3], TA_by30=acc[:, 4], wrong=acc[:, 5])
    return rows


STOCK = {}
if __name__ == "__main__":
    out = {}
    plans = [("netbci", "ses-01", ["ds7g", "ds7lr"]), ("ds7g", "ses-01", ["netbci", "ds7lr"]), ("ds7g", "ses-02", ["netbci", "ds7lr"]), ("ds7lr", "ses-01", ["netbci", "ds7g"])]
    for c, es, coh in plans:
        res = run_corpus(c, es, coh)
        for mode in ("ext", "pub+ext"):
            STOCK.clear()
            rows = evaluate(res, es, mode)
            key = f"{c}|enrol {es}|gallery {mode}"
            out[key] = {n: {k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in r.items()} for n, r in rows.items()}
            print(f"== {key} (T={res['T']}, cohort {coh}, |E|={len(res['people']) // 2}): mean and SD over {NSPLIT} person splits (E/U and swapped)")
            for n, r in out[key].items():
                print(f"{n:11s} TA {r['TA'][0]:.3f}±{r['TA'][1]:.3f}  misID {r['misID'][0]:.3f}  FA {r['FA'][0]:.3f}  wrong {r['wrong'][0]:.3f}  med n {r['med_n'][0]:.0f}", flush=True)
        np.savez_compressed(outp.replace(".json", f"_{c}_{es}.npz"), raw=res["raw"].astype(np.float32), sn=res["sn"].astype(np.float32), asn=res["asn"].astype(np.float32))
    json.dump(out, open(outp, "w"), indent=1)
