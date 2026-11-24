"""Lane G1 constants: DEF = shipped values (S2 DEF plus the coupling keys of the em stage, default off = shipped code path); ARMS = the registered arms."""
DEF = dict(rebal=dict(alpha=1.15, k=5.0, W=30),
           em=dict(lam=0.5, n0=60, m0=20, shrink=0.7, bem=2.0, nmin=20, mode=None, edge=None, ew=12.0),
           tsem=dict(rho=0.7, shrink=0.7, m0=20, n0=60, lam=0.15, bts=4.0, nmin=15), csp=dict(on=1, k=20, q=None, cap=None), count=dict(on=1))
ARMS = {"base": {}, "zero": dict(mode="zero"), "anchor": dict(mode="anchor"), "shared": dict(mode="shared"),
        "cog": dict(mode="edge", edge=[1, 2]), "wrongWM": dict(mode="edge", edge=[2, 0]), "wrongSM": dict(mode="edge", edge=[1, 0])}
