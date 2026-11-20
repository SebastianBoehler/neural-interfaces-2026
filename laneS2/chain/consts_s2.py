"""Lane S2 constants: DEF = shipped values (q, cap None = read from the shipped CSP config); MOVES = the two-level one-constant grid."""
DEF = dict(rebal=dict(alpha=1.15, k=5.0, W=30), em=dict(lam=0.5, n0=60, m0=20, shrink=0.7, bem=2.0, nmin=20),
           tsem=dict(rho=0.7, shrink=0.7, m0=20, n0=60, lam=0.15, bts=4.0, nmin=15), csp=dict(on=1, k=20, q=None, cap=None), count=dict(on=1))
MOVES = dict(  # one-constant moves of the grid (two levels each, a factor 2 or the natural neighbours)
    rebal=dict(alpha=(1.0, 1.3), k=(2.0, 10.0), W=(15, 60)),
    em=dict(lam=(0.3, 0.7), n0=(30, 120), m0=(10, 40), shrink=(0.5, 0.85), bem=(0.0, 4.0), nmin=(10, 30)),
    tsem=dict(rho=(0.5, 0.85), shrink=(0.5, 0.85), m0=(10, 40), n0=(30, 120), lam=(0.10, 0.22), bts=(0.0, 8.0), nmin=(10, 25)),
    csp=dict(on=(0,), k=(10, 40), q=(0.3, 0.7), cap=(0.05, 0.15)), count=dict(on=(0,)))
