"""s229 summary: r15 - s15 (in-run control, tag ttt18<order>LP) on the combination line, new people, per order; the s229 rule."""
import re
from pathlib import Path
D = Path(__file__).resolve().parents[3] / "runs/s229"
ORD = [f"within{i}" for i in range(12)] + [f"full{i}" for i in range(4)] + [f"runs{i}" for i in range(4)]


def comb(o, v):
    f = D / f"{o}_{v}.log"
    m = f.exists() and re.search(r"combination over v5\.1b, both shuffled\s+new ([+-][\d.]+) .*without P10 ([+-][\d.]+)", f.read_text())
    return (float(m.group(1)), float(m.group(2))) if m else None


d = {o: (r[0] - c[0], r[1] - c[1], c[0]) for o in ORD if (c := comb(o, "s15")) and (r := comb(o, "r15"))}
for o in ORD:
    print(f"{o:9s} " + (f"s15 {d[o][2]:+.2f}  r15 - s15 {d[o][0]:+.2f} (without P10 {d[o][1]:+.2f})" if o in d else "-"))
if d:
    x = [v[0] for v in d.values()]; fu = [d[o][0] for o in d if o.startswith("full")]; mean, share = sum(x) / len(x), sum(v >= 0 for v in x) / len(x)
    by = {s: [d[o][0] for o in d if o.startswith(s)] for s in ("within", "full", "runs")}
    print("scheme means: " + "  ".join(f"{s} {sum(v) / len(v):+.2f} (n={len(v)})" for s, v in by.items() if v))
    fm = sum(fu) / len(fu) if fu else float("nan")
    verdict = "CANDIDATE" if mean >= 0.1 and share >= 0.6 and fm >= 0 else ("CLOSED" if mean <= -0.3 else "UNDECIDED")
    print(f"mean {mean:+.2f} over {len(x)} orders, {share:.0%} >= 0, full mean {fm:+.2f} -> s229 rule: {verdict}" + ("" if len(x) == len(ORD) else " (incomplete)"))
