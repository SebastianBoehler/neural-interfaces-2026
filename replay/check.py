"""Check that the replay reproduces the reference dev scores of the packaged chain (routing v3).
Usage: python replay/check.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env  # noqa: E402

REF = {"B2_pool": 80.0131, "B2_s345": 79.8474, "B2_s012": 80.1446, "B2_s0": 79.1770, "B2_s1": 79.7214, "B2_s2": 80.2093}
ok = True
for bn in env.BASES:
    s = env.A.summary(env.A.system(bn, env.A.FAMS)); j = 0.5 * s["dn"].mean() + 0.5 * s["ds"].mean()
    ok &= abs(j - REF[bn]) < 5e-5
    print(f"{bn:8s} J {j:.4f} (reference {REF[bn]:.4f}) new {s['dn'].mean():.2f} seen {s['ds'].mean():.2f}")
print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
