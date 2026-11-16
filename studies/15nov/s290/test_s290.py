"""s290 unit checks without data: band sets, coupling/hierarchy identities. Run: python -I studies/15nov/s290/test_s290.py"""
import os, sys
os.environ.setdefault("NI26_ROOT", os.path.abspath(__file__ + "/../../../.."))
sys.path[:0] = [os.path.dirname(__file__)]
import numpy as np
import s290 as S
print("MB", [S.OV17[i] for i in S.MB]); print("TA", [S.OV17[i] for i in S.TA]); print("THA", [S.OV17[i] for i in S.THA])
p = S.hier(np.array([0.2, 0.9]), np.array([0.3, 0.5])); assert np.allclose(p.sum(1), 1) and np.allclose(p[0], [0.2, 0.24, 0.56])
r = {(0, 1): np.array([0.5, 0.9]), (0, 2): np.array([0.5, 0.9]), (1, 2): np.array([0.5, 0.5])}
c = S.couple(r); assert np.allclose(c.sum(1), 1) and np.allclose(c[0], 1 / 3) and c[1].argmax() == 0 and np.isclose(c[1, 1], c[1, 2])
t = np.array([0.7, 0.2, 0.1]); q = {(0, 1): np.array([.7 / .9]), (0, 2): np.array([.7 / .8]), (1, 2): np.array([.2 / .3])}
print("coupling of consistent pairwise", S.couple(q)[0], "true", t); assert S.couple(q)[0].argmax() == 0
print("OK")
