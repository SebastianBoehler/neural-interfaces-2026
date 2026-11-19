"""NI-V report: absolute score per cohort, paired delta A-B with people-nested bootstrap, decisions changed, returning
dose, zero-routing check, null calibration, sealed-weighted combination.

python collect.py RELEASE_OUT_DIR OUT.json --a TAG --b TAG [--null TAG] [--cohort-a cohort_a.npz] [--data windows.npz]
TAG = package tag (out/TAG for B-new, out/ret_TAG_k{0..8} and out/fa_TAG_k{0..8} for the returning arm).
Headline = full-session shuffle (fresh orders); within-run is secondary. Cohort A exists for the legacy orders
full0 / within0 only (see cohort_a.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank  # noqa: E402
import stats  # noqa: E402

NEW_W = 0.57


def cohort_b_new(out, tag, data):
    return bank.assemble(Path(out) / tag, data)


def cohort_b_ret(out, tag, data):
    """Visit 2 of person k scored on package k (visit 1 enrolled); missing cells stay NaN."""
    full = None
    for k in range(9):
        d = Path(out) / f"ret_{tag}_k{k}"
        if not d.exists() or not list(d.glob("p*_c*.npz")):
            continue
        r = bank.assemble(d, data)
        if full is None:
            full = {key: np.full_like(r[key], np.nan) if key in ("cell", "rec", "routed") else r[key].copy() for key in r}
        for key in ("cell", "rec", "routed"):
            full[key][:, :, k, 1] = r[key][:, :, k, 1]
        full["pred"] = np.where(r["pred"] >= 0, r["pred"], full["pred"]) if "pred" in full else r["pred"]
    return full


def false_recognition(out, tag, data):
    """Streams of the 8 other people through package k: how many recognise (rec >= 0 or routed > 0)."""
    n = hits = 0
    per = {}
    for k in range(9):
        d = Path(out) / f"fa_{tag}_k{k}"
        if not d.exists() or not list(d.glob("p*_c*.npz")):
            continue
        r = bank.assemble(d, data)
        m = np.isfinite(r["cell"][0, 0])  # order full:0 only
        m[k] = False
        rec = (r["rec"][0, 0] >= 0) | (r["routed"][0, 0] > 0)
        per[k] = int((rec & m).sum())
        n += int(m.sum())
        hits += per[k]
    return dict(streams=n, recognised=hits, per_package=per)


def recognition(ret):
    rec = (ret["rec"] >= 0) | (ret["routed"] > 0)
    ok = np.isfinite(ret["cell"][:, :, :, 1])
    return dict(visit2_streams=int(ok.sum()), recognised=int((rec[:, :, :, 1] & ok).sum()),
                persons_recognised_any=int((rec[:, :, :, 1] & ok).any((0, 1)).sum()),
                per_person_rate=np.round((rec[:, :, :, 1] & ok).sum((0, 1)) / np.maximum(ok.sum((0, 1)), 1), 3).tolist())


def absolute(cell):
    t = stats.person_table(cell)
    return dict(full=float(np.nanmean(t[0])), within=float(np.nanmean(t[1])))


def boot_draws(cell_a, cell_b, scheme, seed):
    d = (stats.person_table(cell_a) - stats.person_table(cell_b))[[scheme]]
    return stats.boot(d, np.random.default_rng(seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("json")
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--null")
    ap.add_argument("--cohort-a"); ap.add_argument("--data", default="data/scherer2015/windows.npz")
    a = ap.parse_args()
    y = np.load(a.data, allow_pickle=False)["y"]
    res = {"A_tag": a.a, "B_tag": a.b, "headline": "full-session shuffle (fresh orders 71000-71007)"}
    na, nb = cohort_b_new(a.out, a.a, a.data), cohort_b_new(a.out, a.b, a.data)
    res["scherer_new"] = dict(absolute={a.a: absolute(na["cell"]), a.b: absolute(nb["cell"])},
                              delta=stats.report(na["cell"], nb["cell"]),
                              decisions={s: stats.changed(na["pred"][i], nb["pred"][i], y) for i, s in enumerate(("full", "within"))},
                              recognised_new=recognition(na))
    if a.null:
        n2 = cohort_b_new(a.out, a.null, a.data)
        res["null"] = dict(tag=a.null, full=stats.null_summary(na["cell"], n2["cell"]),
                           within=stats.null_summary(na["cell"], n2["cell"], schemes=stats.SECONDARY),
                           decisions_full=stats.changed(na["pred"][0], n2["pred"][0], y))
    draws_new = [(boot_draws(na["cell"], nb["cell"], 0, 31), 9)]
    if a.cohort_a and (a.a, a.b) == ("cap250", "v51b"):  # cohort A replays exist only for this contrast
        z = np.load(a.cohort_a)
        res["cohort_A"] = {}
        for si, scheme in enumerate(z["scheme"]):
            ca, cb = z["cap250"][si][None, None], z["incumbent"][si][None, None]  # (1,1,10,6): scheme/order axes of size 1
            t = np.nanmean(ca, 3)[0] - np.nanmean(cb, 3)[0]
            b = stats.boot(np.repeat(t[None], 1, 0), np.random.default_rng(37))
            res["cohort_A"][str(scheme)] = dict(abs_cap250=float(np.nanmean(ca.mean(3))), abs_incumbent=float(np.nanmean(cb.mean(3))),
                                                pp=float(t.mean()), i80=stats.interval(b, .8), i95=stats.interval(b, .95), n_orders=1,
                                                by_person=t[0].tolist())
            if scheme == "full":
                draws_new.append((b, 10))
    ra, rb = cohort_b_ret(a.out, a.a, a.data), cohort_b_ret(a.out, a.b, a.data)
    if ra is not None and rb is not None and np.isfinite(ra["cell"]).any() and np.isfinite(rb["cell"]).any():
        res["scherer_returning"] = dict(
            absolute={a.a: absolute(ra["cell"]), a.b: absolute(rb["cell"])}, delta=stats.report(ra["cell"], rb["cell"]),
            recognition={a.a: recognition(ra), a.b: recognition(rb)},
            dose_enrolled_minus_base={a.a: stats.report(ra["cell"], na["cell"]), a.b: stats.report(rb["cell"], nb["cell"])},
            false_recognition={a.a: false_recognition(a.out, a.a, a.data), a.b: false_recognition(a.out, a.b, a.data)})
        ret_draws = boot_draws(ra["cell"], rb["cell"], 0, 41)
        w = np.array([n for _, n in draws_new], float)
        new_draws = sum(wi * d for wi, (d, _) in zip(w / w.sum(), draws_new))
        res["sealed_weighted"] = dict(weights=dict(new=NEW_W, returning=1 - NEW_W),
                                      new_cohorts="Scherer-new (9) + PROTEUS held-new full0 (10, one order)" if len(draws_new) > 1 else "Scherer-new (9)",
                                      combined=stats.combine_boot(new_draws, ret_draws, NEW_W))
    Path(a.json).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k in ("scherer_new", "null")}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
