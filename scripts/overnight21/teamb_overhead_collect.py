"""Summarise e2e-*.json timing jobs: python teamb_overhead_collect.py OUT_DIR JOBID... ; prints per job and per GPU the ms/window and projections."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
by_gpu = {}
for job in sys.argv[2:]:
    d = json.loads((out / f"e2e-{job}.json").read_text())
    ref = next(k for k in d if isinstance(d[k], dict) and "added_vs_ref_ms" in d[k] and d[k]["added_vs_ref_ms"] == 0.0)
    print(job, d["gpu"], "ref", ref)
    for k, v in d.items():
        if isinstance(v, dict) and "projected_s" in v:
            print(f"  {k:70s} {v['ms_per_window']:7.1f} ms/win  +{v['added_vs_ref_ms']:6.1f}  proj {v['projected_s']:6.0f} s  inside {({a: round(b, 1) for a, b in v['inside_ms_per_window'].items()})}")
            by_gpu.setdefault((d["gpu"], k), []).append((v["added_vs_ref_ms"], v["projected_s"]))
print()
for (gpu, k), v in sorted(by_gpu.items()):
    if all(a == 0 for a, _ in v):
        continue
    print(f"{gpu:28s} {k:70s} n={len(v)} proj {min(p for _, p in v):.0f}-{max(p for _, p in v):.0f} s  added {min(a for a, _ in v):.1f}..{max(a for a, _ in v):.1f} ms")
