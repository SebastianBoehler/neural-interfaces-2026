"""Successive halving: write the next stage's config file = top FRAC of the non-reference configs by BA + reference arms.
python promote.py LEADERBOARD.json OUT.json [--frac 0.2] [--top N]"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("lb")
ap.add_argument("out")
ap.add_argument("--frac", type=float, default=0.2)
ap.add_argument("--top", type=int, default=0)
a = ap.parse_args()
lb = json.load(open(a.lb))
refs = {"A0", "A1", "A2", "A2h"}
rows = [r for r in lb["rows"] if r["name"] not in refs]
n = a.top or int(round(a.frac * len(rows)))
sel = {r["name"]: r["cfg"] for r in rows[:n]}
sel.update({r["name"]: r["cfg"] for r in lb["rows"] if r["name"] in refs})
json.dump(sel, open(a.out, "w"), indent=0)
print(len(sel), "configs ->", a.out, "(promoted", n, "of", len(rows), ")")
