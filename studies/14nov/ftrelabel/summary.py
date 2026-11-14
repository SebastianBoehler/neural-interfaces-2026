"""s221 summary: the combination line (new people) of every runs/s221/<order>_<tag>_<arm>.log; each ftrel arm minus the
in-run s15 control (RP s15), per order, within 0-3 mean and full 0/1 mean; RF / KF s15 alongside. Kill rule of s221 for b15."""
import re
from pathlib import Path
D = Path(__file__).resolve().parents[3] / "runs/s221"; ORD = ["within0", "within1", "within2", "within3", "full0", "full1"]
ARMS = [("RP", "r15"), ("RP", "e15"), ("RP", "b15"), ("RP", "q15"), ("RF", "s15"), ("KF", "s15")]


def comb(o, t, v):
    f = D / f"{o}_{t}_{v}.log"
    m = f.exists() and re.search(r"combination over v5\.1b, both shuffled\s+new ([+-][\d.]+) .*without P10 ([+-][\d.]+)", f.read_text())
    return (float(m.group(1)), float(m.group(2))) if m else None


ctl = {o: comb(o, "RP", "s15") for o in ORD}
print("control RP s15 (combination over v5.1b, new; without P10): " + "  ".join(f"{o} {c[0]:+.2f} ({c[1]:+.2f})" if c else f"{o} -" for o, c in ctl.items()))
for t, v in ARMS:
    d = {o: (a[0] - ctl[o][0], a[1] - ctl[o][1]) for o in ORD if (a := comb(o, t, v)) and ctl[o]}
    w = [d[o][0] for o in ORD[:4] if o in d]; fu = [d[o][0] for o in ORD[4:] if o in d]
    row = "  ".join(f"{o} {d[o][0]:+.2f}" if o in d else f"{o} -" for o in ORD)
    wm = sum(w) / len(w) if w else float("nan"); fm = sum(fu) / len(fu) if fu else float("nan")
    print(f"{t} {v} - control: {row} | within mean {wm:+.2f} (n={len(w)}) full mean {fm:+.2f} (n={len(fu)})"
          + (f" | s221 kill rule: {'PASS' if wm >= 0.15 and fm >= 0 else 'KILLED'}" if v == "b15" and len(w) == 4 and len(fu) == 2 else ""))
