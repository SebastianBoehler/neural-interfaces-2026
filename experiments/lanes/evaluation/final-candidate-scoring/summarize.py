"""Lane N: assemble NI-V paired deltas (vs cap250 and vs hardcbB) from out/report_TAG_vs_REF.json into one JSON and a markdown table.
usage (release dir, in the worker container): python laneN/summarize.py OUT_DIR OUT_JSON OUT_MD TAG...  -- missing reports are listed, not skipped silently."""
import json
import sys
from pathlib import Path

out, ojson, omd, tags = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
REFS = ("cap250", "hardcbB")
res, missing = {}, []


def cell(d):
    bp = d["by_person"]
    return dict(pp=d["pp"], i80=d["i80"], worst_person=min(bp), worst_idx=int(bp.index(min(bp))), by_person=bp)


for t in tags:
    for r in REFS:
        if t == r:
            continue
        f = out / f"report_{t}_vs_{r}.json"
        if not f.exists():
            missing.append(f.name)
            continue
        d = json.load(open(f))
        res.setdefault(t, {})[r] = dict(
            b_new_full=cell(d["scherer_new"]["delta"]["primary_full"]), b_new_within=cell(d["scherer_new"]["delta"]["secondary_within"]),
            b_ret_full=cell(d["scherer_returning"]["delta"]["primary_full"]), b_ret_within=cell(d["scherer_returning"]["delta"]["secondary_within"]),
            sealed_weighted=d["sealed_weighted"]["combined"],
            absolute_new=d["scherer_new"]["absolute"], absolute_ret=d["scherer_returning"]["absolute"])
json.dump(dict(missing=missing, results=res), open(ojson, "w"), indent=1)
L = ["| tag | vs | B-new full | B-new within | B-ret full | B-ret within | worst person new/ret (full) | NI-V sealed-weighted |", "|---|---|---|---|---|---|---|---|"]
for t, rr in res.items():
    for r, v in rr.items():
        f = lambda c: f"{c['pp']:+.3f} [{c['i80'][0]:+.2f},{c['i80'][1]:+.2f}]"
        s = v["sealed_weighted"]
        L.append(f"| {t} | {r} | {f(v['b_new_full'])} | {f(v['b_new_within'])} | {f(v['b_ret_full'])} | {f(v['b_ret_within'])} | "
                 f"{v['b_new_full']['worst_person']:+.2f} / {v['b_ret_full']['worst_person']:+.2f} | {s['pp']:+.3f} [{s['i80'][0]:+.2f},{s['i80'][1]:+.2f}] |")
if missing:
    L.append("\nMISSING: " + ", ".join(missing))
open(omd, "w").write("\n".join(L) + "\n")
print("\n".join(L))
