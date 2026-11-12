"""Integration check of the v5.2 package (personal SessionBN traces on routed windows) against v5.1b, public windows.

Usage: python scripts/check_v52.py PKG_V52 PKG_V51B [--out JSON]
Main pass (all 14 public sessions, one window per call, both packages):
  (1) the shadow tail (gate input) equals v5.1b on every window and the routed output equals v5.1b on every window
      before the session's first routed window; the gate routes the same windows (v5.1b's own-stage counter);
  (2) every routed window is swapped (no errors) and has a personal record;
  (3) on routed windows the personal posteriors equal the personal traces (fresh torch.jit.load) run on the session's
      network inputs recomputed by fresh input streams (< 1e-4), the buffered inputs equal those inputs, and the
      routed tail's mix equals mix + sum_m w_m (reference - base) (< 1e-4);
  (4) not a no-op: the routed outputs differ from v5.1b on some windows.
Forced-off pass (three sessions, both gates' tau set to +inf in both packages): outputs equal on every window, no
personal call. Switch pass (two sessions): the person is forced to alternate between two people with gaps, so traces
are created lazily for several people and caught up after lagging; posteriors must still match the reference.
"""
import copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
from add_rebalance import load_solver  # noqa: E402

TOL = 1e-4


def run(m, d, sessions, sched=None):
    """Stream sessions window by window. Returns per-window routed outputs and shadow inputs, the route logs per
    session and (v5.2) per session the personal records [(t, j, P, mix, base, routed mix)] and buffered inputs."""
    rec = dict(out=[], shadow=[], logs=[], pn=[], buf=[])
    step0 = m.route.step

    def step(x, u, mix, core1, qs):
        o = step0(x, u, mix, core1, qs); rec["out"].append(np.array(o)); rec["shadow"].append(np.array(qs)); return o
    m.route.step = step
    mixes = []
    if not hasattr(m, "pnets"):
        m.pnets = None  # v5.1b
    if m.pnets is not None:
        pm0 = m._pmix

        def pmix(j, b, t, pm):
            jj = sched(t, j) if sched else j
            o = pm0(jj, b, t, pm)
            mixes.append((pm.copy(), np.stack([m.members[i][b] for i in m.pnets.idx]), o)); return o
        m._pmix = pmix
    for s in sessions:
        m.reset_state(); rows = np.flatnonzero(d["stream"] == s)
        if m.pnets is not None:
            m.pnets.record = []; mixes.clear()
        for i in rows:
            m.predict(d["X"][i:i + 1])
        rec["logs"].append(dict(m.route.log))
        if m.pnets is not None:
            assert len(m.pnets.record) == len(mixes)
            rec["pn"].append([(t, j, P) + mx for (t, j, P), mx in zip(m.pnets.record, mixes)])
            rec["buf"].append({k: np.stack(v) for k, v in m.pnets.buf.items()})
    m.route.step = step0  # unwrap, so a later pass records only into its own lists
    if m.pnets is not None:
        m._pmix, m.pnets.record = pm0, None
    return rec


@torch.no_grad()
def reference(m, d, rows, j, upto, pkg):
    """Person j's traces, freshly loaded, on network inputs recomputed by fresh input streams: (T, M, 3)."""
    streams = {}
    for k in set(m.pnets.keys):
        st = copy.deepcopy(m.groups[k][0]); st.reset_state()
        streams[k] = np.stack([st(np.nan_to_num(np.asarray(d["X"][i], np.float32))) for i in rows[:upto]])
    out = []
    for mi, k in enumerate(m.pnets.keys):
        mod = torch.jit.load(str(pkg / str(m.pnets_files[mi, j])), map_location="cpu").eval()
        for name, b in mod.named_buffers():
            if name.rsplit(".", 1)[-1] in ("s1", "s2", "n"):
                b.zero_()
        out.append(torch.softmax(mod(torch.from_numpy(streams[k])).double(), 1).numpy())
    return np.stack(out, 1), streams


def compare(m, d, sessions, rec, pkg):
    """Max differences of the personal posteriors, buffered inputs and routed mixes against the reference."""
    dp = dx = dm = 0.0; n = 0; w = m.pnets.w
    for s, g, buf in zip(sessions, rec["pn"], rec["buf"]):
        if not g:
            continue
        rows = np.flatnonzero(d["stream"] == s); upto = max(r[0] for r in g) + 1
        for j in sorted({r[1] for r in g}):
            R, streams = reference(m, d, rows, j, upto, pkg)
            for k, X in streams.items():
                dx = max(dx, float(np.abs(buf[k][:upto] - X).max()))
            for t, jj, P, pm, base, o in g:
                if jj == j:
                    dp = max(dp, float(np.abs(P - R[t]).max())); n += 1
                    dm = max(dm, float(np.abs(o - (pm + (w[:, None] * (R[t] - base)).sum(0))).max()))
    return dict(max_posterior_diff=dp, max_input_diff=dx, max_mix_diff=dm, routed_windows_compared=n)


def main():
    v52, v51 = Path(sys.argv[1]), Path(sys.argv[2]); d = protocol.load(); sessions = list(dict.fromkeys(d["stream"]))
    t0 = time.time(); m = load_solver(v52, "v52"); load_s = time.time() - t0
    assert m.pnets is not None and len(m.pnets.idx) == 30 and sum(map(len, m.pnets.pristine)) == 300
    m.pnets_files = np.load(v52 / "weights.npz")["pnet_files"]
    res = dict(load_seconds_v52=round(load_s, 1))
    # main pass
    t0 = time.time(); new = run(m, d, sessions); res["stream_seconds_v52"] = round(time.time() - t0, 1)
    b = load_solver(v51, "v51b"); t0 = time.time(); old = run(b, d, sessions); res["stream_seconds_v51b"] = round(time.time() - t0, 1)
    Yn, Yo, Sn, So = (np.asarray(x) for x in (new["out"], old["out"], new["shadow"], old["shadow"]))
    res["shadow_equal_every_window"] = bool(np.array_equal(Sn, So))
    pre_eq, sess, names = True, {}, [str(q) for q in np.load(v52 / "weights.npz")["route_names"]]
    cut = np.cumsum([0] + [int((d["stream"] == s).sum()) for s in sessions])
    for k, s in enumerate(sessions):
        lgn, lgo, g, a, e = new["logs"][k], old["logs"][k], new["pn"][k], cut[k], cut[k + 1]
        assert lgn["proutes"] == lgo["routed"] == lgn["routed"] == len(g), (s, lgn, lgo, len(g))
        assert lgn["pswap"] == lgn["proutes"] and lgn["perr"] == 0, (s, lgn)
        f = g[0][0] if g else e - a
        pre_eq &= bool(np.array_equal(Yn[a:a + f], Yo[a:a + f]))
        sess[s.split("/")[-1]] = dict(windows=int(e - a), routed=lgn["proutes"], swapped=lgn["pswap"], errors=lgn["perr"],
                                      first_routed=int(f) if g else None, people=sorted({names[r[1]] for r in g}),
                                      labels_changed=int((Yn[a:e].argmax(1) != Yo[a:e].argmax(1)).sum()),
                                      max_out_diff=float(np.abs(Yn[a:e] - Yo[a:e]).max()))
    routed_sessions = [s for k, s in enumerate(sessions) if new["pn"][k]]
    res["equal_before_first_routed_and_unrouted_sessions"] = pre_eq
    res["unrouted_sessions"] = len(sessions) - len(routed_sessions)
    res["main"] = compare(m, d, sessions, new, v52)
    y = d["y"]; res["acc_v52"] = float(np.mean(Yn.argmax(1) == y)); res["acc_v51b"] = float(np.mean(Yo.argmax(1) == y))
    res["labels_changed"] = int((Yn.argmax(1) != Yo.argmax(1)).sum()); res["max_out_diff"] = float(np.abs(Yn - Yo).max())
    res["sessions"] = sess
    # forced-off pass
    off = sessions[:3]
    for mm in (m, b):
        mm.route.tau = mm.route.bio.tau = np.inf
    fo, fb = run(m, d, off), run(b, d, off)
    res["forced_off_equal"] = bool(np.array_equal(np.asarray(fo["out"]), np.asarray(fb["out"])))
    res["forced_off_personal_calls"] = sum(map(len, fo["pn"]))
    # switch pass: restore the gates; alternate two people with gaps
    z = np.load(v52 / "weights.npz")
    m.route.tau, m.route.bio.tau = float(z["route_tau"]), float(z["bio_tau"])
    sw = [s for s in routed_sessions if s in (sessions[0], sessions[10])]
    sched = lambda t, j: (j if (t // 7) % 3 == 0 else (j + 3) % 10 if (t // 7) % 3 == 1 else (j + 5) % 10)
    rs = run(m, d, sw, sched)
    res["switch"] = compare(m, d, sw, rs, v52)
    res["switch"]["people_used"] = sorted({int(r[1]) for g in rs["pn"] for r in g})
    print(json.dumps(res, indent=1))
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(res, indent=1))
    ok = (res["shadow_equal_every_window"] and pre_eq and res["forced_off_equal"] and res["forced_off_personal_calls"] == 0
          and res["main"]["max_posterior_diff"] < TOL and res["main"]["max_mix_diff"] < TOL and res["main"]["max_input_diff"] == 0
          and res["switch"]["max_posterior_diff"] < TOL and res["switch"]["max_mix_diff"] < TOL and len(res["switch"]["people_used"]) >= 3
          and res["main"]["routed_windows_compared"] == sum(v["swapped"] for v in sess.values()) and res["max_out_diff"] > 0)
    assert ok, "v5.2 check FAILED"
    print("V52 CHECK OK")


if __name__ == "__main__":
    main()
