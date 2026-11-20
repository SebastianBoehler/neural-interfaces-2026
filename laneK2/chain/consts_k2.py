"""Lane K2 constants: DEF = shipped values of the S2 stages (q, cap None = read from the shipped CSP config); em.prior None = stock responsibilities."""
DEF = dict(rebal=dict(alpha=1.15, k=5.0, W=30), em=dict(lam=0.5, n0=60, m0=20, shrink=0.7, bem=2.0, nmin=20, prior=None),
           tsem=dict(rho=0.7, shrink=0.7, m0=20, n0=60, lam=0.15, bts=4.0, nmin=15), csp=dict(on=1, k=20, q=None, cap=None), count=dict(on=1))
PRIORS = {"uniform": None, "feas": 0.2, "feas50": 0.5, "oracle": None}  # em.prior value -> delta of the feasible set (None: no feasible set)
SCALES = (0.5, 1.0, 2.0)
W4, CLIP4, W6 = 0.15, 0.5, 0.10  # spec weights: idea 4 (x scale, clipped), idea 6 (x scale)
