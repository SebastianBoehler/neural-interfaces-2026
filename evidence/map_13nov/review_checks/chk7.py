import sys, os
sys.dont_write_bytecode = True
import numpy as np
REPO = "<repo>"
F = REPO + "/studies/12nov/inpop3.py"; __file__ = F
src = open(F).read().split("\nfor bn in (\"B2_pool\"")[0]
exec(compile(src, F, "exec"))
for bn in ("B2_pool", "B2_s345", "B2_s012"):
    PF.clear(); PF.update(BASE_PF); P0 = chain(bn, True); r0 = ba_people(P0); st = ba_people(staged(P0)) - r0
    print(f"{bn}: in-population chain {r0.mean():.1f} | stage {st.mean():+.2f}", flush=True)
    for v in ("pfh60",):
        for tag, keep in (("as scored (slots 5-8 and the shallow slots 3, 4 personal)", ()), ("shallow slots 3, 4 not personal (package: no personal shallow nets)", ("sh",)),
                          ("only the shallow slots 3, 4 personal", ("fbraw", "fb", "sinc", "fbl"))):
            PF.clear(); PF.update(PFV[v]); PF.update({a: BASE_PF[a] for a in keep}); P1 = chain(bn, True); a = ba_people(P1) - r0; b = ba_people(staged(P1)) - r0; d = b - st
            bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
            print(f"   {v} {tag}: nets alone {a.mean():+.2f} | against the stage alone {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] ahead {(d > 0).sum()}/10", flush=True)
    PF.clear(); PF.update(BASE_PF)
