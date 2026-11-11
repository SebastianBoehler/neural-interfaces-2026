"""Entry point of the dev chain replay: `sys.path.insert(0, REPO + "/replay"); import env; A = env.A`.

A.system(base, A.FAMS) replays the packaged chain (member mix -> rebalancing -> GEB SessionEM -> GEB TSEM -> routing
v3) on one dev base; A.summary(...) gives per-person balanced accuracy. Reference J (0.5 new + 0.5 seen):
B2_pool 80.0131, B2_s345 79.8474, B2_s012 80.1446, B2_s0 79.1770, B2_s1 79.7214, B2_s2 80.2093 (python replay/check.py).
Routing v4.1 (the uploaded package) is replayed by studies/11nov/rt4/route4.py on top of this."""
import paths  # noqa: F401  (sets sys.path)
import grid  # noqa: E402,F401  (loads the dev inputs)
import alib as A  # noqa: E402

SC = A.SC
BASES = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")
EXPL, FRESH = BASES[:2], BASES[2:]
PEOPLE, SEEN, G1, G2 = SC.PEOPLE, SC.SEEN, SC.G1, SC.G2
