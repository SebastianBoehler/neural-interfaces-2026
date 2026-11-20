"""Lane FT-labels collector: flipped (v6_ft_labels=1) arms minus their unflipped references, from scores.json / predictions.npz only.
usage: ftl_collect.py PROD_RUN OV_RUN FT_RUN OUT.md [b6s]   (b6s: lane FT-B6s contrasts X-stock, C-stock, X-C from FT_RUN/o<chunk>/{X,C})
References: orders 71000-71003 = PROD_RUN/{stock,hardcb}; 71004+ = OV_RUN/o<chunk>/{stock,hardcb_safe}. Flipped: FT_RUN/o<chunk>/{stock,hardcb_safe,hardcb_teach}.
Contrasts: S = flipped stock - stock; H1 = flipped hardcb_safe - hardcb; H2 = flipped hardcb_teach - hardcb; H2-H1.
Intervals (80 %, seed 0, 10,000 draws): people only (orders fixed, people resampled), and people-nested (people, then orders). New people = P01-P10 held-new
cell-mean; returning P04/P10 from ret|order by_person (separately; neither decides). J = 0.57 new + 0.43 (0.2 P04 + 0.8 P10) is an UNVERIFIED assumption
(the eight other returning people count 0). Changed decisions per new-person fold from predictions.npz; a new-person fold with 0 changed fails loudly."""
import json
import sys
from pathlib import Path

import numpy as np

prod, ov, ft, out = (Path(a) for a in sys.argv[1:5])
rng = np.random.default_rng(0)
PEOPLE = [f"P{i:02d}" for i in range(1, 11)]


def chunk(o):
    return 71000 if o < 71004 else 71004 + 4 * ((o - 71004) // 4)


def ref_dir(o, arm):  # arm: stock | hardcb | ft:<arm> (another arm of FT_RUN)
    if arm.startswith("ft:"):
        return ft_dir(o, arm[3:])
    if o < 71004:
        return prod / arm
    return ov / f"o{chunk(o)}" / ("stock" if arm == "stock" else "hardcb_safe")


def ft_dir(o, arm):
    return ft / f"o{chunk(o)}" / arm


cache = {}


def scores(d):
    if d not in cache:
        cache[d] = json.load(open(d / "scores.json")) if (d / "scores.json").exists() else None
    return cache[d]


def changed(o, d1, d0):
    """per new-person fold: (changed, delivered) between predictions of d1 and d0 for order o."""
    res = {}
    with np.load(d1 / "predictions.npz") as a, np.load(d0 / "predictions.npz") as b:
        for p in PEOPLE:
            x, y = a[f"new|{o}|{p}"], b[f"new|{o}|{p}"]
            m = (x >= 0) & (y >= 0)
            assert (x >= 0).sum() == (y >= 0).sum() == m.sum(), (o, p)
            res[p] = (int((x[m] != y[m]).sum()), int(m.sum()))
    return res


B6S = len(sys.argv) > 5 and sys.argv[5] == "b6s"
CON = ({"X-stock": ("X", "stock", "stock"), "C-stock": ("C", "stock", "stock"), "X-C": ("X", "ft:C", "C")} if B6S else
       {"S": ("stock", "stock", "stock"), "H1": ("hardcb_safe", "hardcb", "hardcb_safe"), "H2": ("hardcb_teach", "hardcb", "hardcb_safe")})
ARMS = ("X", "C") if B6S else ("stock", "hardcb_safe", "hardcb_teach")
lines = (["# Lane FT-B6s: Sinc family on the B6 extra-people trunk (X) vs re-pretrained control (C), stock fold packages (pp)", "",
          "X = m7r + ds007221g/lr + netbci Sinc trunk; C = shipped m7r recipe re-run. New people: ten held-new folds, official cell mean. "
          "J uses the UNVERIFIED 57/43 assumption with the eight other returning people counted 0.", ""] if B6S else
         ["# Lane FT-labels: v6_ft_labels flipped, per-fold packages (pp, flipped arm minus unflipped reference)", "",
          "S = stock + flag; H1 = hardcb_safe + flag (guard clause dropped, teacher = stock mix); H2 = hardcb_safe + flag + hardcb-mix teacher. "
          "New people: ten held-new PROTEUS folds, official cell mean. J uses the UNVERIFIED 57/43 assumption with the eight other returning people counted 0.", ""])
orders_all = sorted({int(k.split("|")[1]) for d in sorted(ft.glob("o*")) for a in ARMS[:1] if (d / a / "scores.json").exists()
                     for k in json.load(open(d / a / "scores.json")) if k.startswith("new|")})
missing = sorted(d.name for d in ft.glob("o*") if not all((d / a / "scores.json").exists() for a in ARMS))
lines.append(f"FT chunks complete for all arms: {[c for c in sorted(d.name for d in ft.glob('o*')) if c not in missing]}; incomplete: {missing or 'none'}.")
summary = {}


def boot(M, mode):  # M orders x people
    n, m = M.shape
    if mode == "people":
        v = [M[:, rng.integers(0, m, m)].mean() for _ in range(10000)]
    else:
        v = [M[rng.integers(0, n, n)][:, rng.integers(0, m, m)].mean() for _ in range(10000)]
    return np.percentile(v, [10, 90])


def contrast(name, a_arm, ref_arm_prod, ref_arm_ov):
    rows, ret = [], []
    for o in orders_all:
        fd, rd = ft_dir(o, a_arm), ref_dir(o, ref_arm_prod)
        sf, sr = scores(fd), scores(rd)
        if sf is None or sr is None or f"new|{o}" not in sf or f"new|{o}" not in sr:
            continue
        ch = changed(o, fd, rd)
        zero = [p for p, (c, n) in ch.items() if c == 0]
        rows.append((o, [(sf[f"new|{o}"]["by_person"][p] - sr[f"new|{o}"]["by_person"][p]) for p in PEOPLE],
                     (sf[f"new|{o}"]["cell_mean"] - sr[f"new|{o}"]["cell_mean"]), ch, zero,
                     {p: (sf[f"ret|{o}"]["by_person"][p] - sr[f"ret|{o}"]["by_person"][p]) for p in ("P04", "P10")} if f"ret|{o}" in sf and f"ret|{o}" in sr else None))
    if not rows:
        lines.extend(["", f"## {name}: no complete order yet"]); return
    D = np.array([r[1] for r in rows]); cell = np.array([r[2] for r in rows])
    tot = {p: sum(r[3][p][0] for r in rows) for p in PEOPLE}; den = {p: sum(r[3][p][1] for r in rows) for p in PEOPLE}
    zero_all = [p for p in PEOPLE if tot[p] == 0]
    R = np.array([[r[5]["P04"], r[5]["P10"]] for r in rows if r[5]])
    ps, po = boot(D, "people"), boot(D, "nested")
    lines.extend(["", f"## {name}: {a_arm} minus {ref_arm_prod}" + ("" if B6S else " (flag 1 vs 0)"), "",
              f"- {len(rows)} orders ({rows[0][0]}..{rows[-1][0]}). New people cell mean **{cell.mean():+.3f}**; people-only paired 80 % [{ps[0]:+.2f}, {ps[1]:+.2f}]; "
              f"people-nested 80 % [{po[0]:+.2f}, {po[1]:+.2f}]; positive in {int((cell > 0).sum())}/{len(cell)} orders; per-order: " + ", ".join(f"{c:+.2f}" for c in cell),
              "- per person (mean over orders): " + ", ".join(f"{p} {D[:, j].mean():+.2f}" for j, p in enumerate(PEOPLE)),
              "- changed decisions per new-person fold, all orders pooled: " + ", ".join(f"{p} {100 * tot[p] / den[p]:.2f} %" for p in PEOPLE)])
    if zero_all:
        lines.append(f"- **FAIL: no changed decision in folds {zero_all} (arm not reaching the code path)**")
    if len(R):
        r04, r10 = R[:, 0].mean(), R[:, 1].mean()
        j = 0.57 * cell.mean() + 0.43 * (0.2 * r04 + 0.8 * r10)
        lines.append(f"- returning ({len(R)} orders) P04 {r04:+.2f}, P10 {r10:+.2f} (neither decides); J (UNVERIFIED 0.57 new + 0.43 x [0.2 P04 + 0.8 P10]) {j:+.3f}")
    summary[name] = dict(mean=float(cell.mean()), people80=ps.tolist(), nested80=po.tolist(), orders=len(rows))
    lines.append(f"- registered rule: upper (people-nested) {po[1]:+.2f} -> " + ("REFUTED (<= 0)" if po[1] <= 0 else
                 ("SUPPORTED (lower > 0 and mean >= +0.2)" if po[0] > 0 and cell.mean() >= 0.2 else "undecided")))
    return D, cell


res = {k: contrast(k, *v) for k, v in CON.items()}
if B6S:
    res = {}
# H2 - H1 on the same orders: difference of the two flipped arms
if not B6S and res["H1"] and res["H2"]:
    D1, D2 = res["H1"][0], res["H2"][0]
    n = min(len(D1), len(D2))
    D = D2[:n] - D1[:n]
    ps, po = boot(D, "people"), boot(D, "nested")
    lines += ["", "## H2 minus H1 (teacher with the hardcb mix vs the stock-mix teacher, both flipped)", "",
              f"- {n} orders: mean {D.mean():+.3f}; people-only 80 % [{ps[0]:+.2f}, {ps[1]:+.2f}]; people-nested [{po[0]:+.2f}, {po[1]:+.2f}]"]
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
(out.with_suffix(".json")).write_text(json.dumps(summary, indent=1))
