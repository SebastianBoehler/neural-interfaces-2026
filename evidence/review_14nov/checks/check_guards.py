"""Isolated checks of lean/src62g or src62h helpers (no weights, no nets, no data). Run: .venv/bin/python evidence/review_14nov/checks/check_guards.py [src62g|src62h]"""
import ast, sys, time, types
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TREE = sys.argv[1] if len(sys.argv) > 1 else "src62h"
SRC = ROOT / "lean" / TREE

# --- 1. which statements of step62 / step61 sit inside a try block (AST, modular source and bundle) ---
for path in (SRC / "v6chain.py", ROOT / f"submissions/lean_v{TREE[3:]}/submission.py"):
    tree = ast.parse(path.read_text())
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Ensemble6"):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("step61", "step62")):
            tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
            raises = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Raise)]
            covered = {c.lineno for t in tries for b in t.body for c in ast.walk(b) if hasattr(c, "lineno")}
            print(f"{path.name}:{fn.name}: try blocks {len(tries)}, raise at lines {raises}, raises inside a try: {[r for r in raises if r in covered]}")

# --- 2. FineTune.step schedule, cap and budget, with the nets stubbed out ---
sys.path.insert(0, str(ROOT / "lean"))
pkg = types.ModuleType("tree"); pkg.__path__ = [str(SRC)]; sys.modules["tree"] = pkg
import importlib
ft_mod = importlib.import_module("tree.finetune")


def make(cap=400, budget=2700.0, t0=20, every=10, on=True):
    f = ft_mod.FineTune.__new__(ft_mod.FineTune)
    f.on, f.cap, f.budget, f.t0, f.every = on, cap, budget, t0, every
    f.nets, f.member, f.X, f.lab, f.wt = {}, {}, {}, [], []
    f.refits, f.steps, f.dtheta, f.capped, f.frozen = 0, 0, 0.0, 0, 0
    f.calls = []
    f.refit = lambda t: f.calls.append(t)
    return f


out = np.array([0.2, 0.5, 0.3])
ft_mod.ft_clock_start()
f = make()
for _ in range(1000):
    f.step({}, {}, out)
print("refits at", f.calls[:3], "...", f.calls[-3:], "count", len(f.calls), "capped", f.capped, "largest prefix", max(f.calls))

# budget: clock never started (a caller that builds Ensemble6 without Solver.load_model, or a re-imported module)
ft_mod.FT_CLOCK.update(t0=None, stopped=False)
f = make(budget=1e-9)
for _ in range(60):
    f.step({}, {}, out)
print("clock never started, budget 1e-9 s: refits", len(f.calls), "frozen", f.frozen, "-> guard silently inert" if f.calls else "")

# budget: started, tiny budget
ft_mod.ft_clock_start(); time.sleep(0.01)
f = make(budget=1e-3)
for _ in range(60):
    f.step({}, {}, out)
print("clock started, budget 1 ms: refits", len(f.calls), "frozen", f.frozen, "lab", len(f.lab), "wt", len(f.wt))

# relabel after a frozen stretch and with fine-tuning off (len(lab) must equal the number of windows so far)
f.relabel([out] * 60); print("relabel after frozen windows: ok")
f = make(on=False)
for _ in range(15):
    f.step({}, {}, out)
f.relabel([out] * 15); print("relabel with v6_ft_on = 0: ok; wt length", len(f.wt))

# schedule key values that check_package does not validate
for bad in (dict(every=0),):
    ft_mod.ft_clock_start()
    f = make(**bad)
    try:
        for _ in range(25):
            f.step({}, {}, out)
        print(bad, "ok")
    except Exception as e:
        print(bad, "->", repr(e))

# --- 3. numpy behaviour the SessionEM fallback relies on ---
try:
    r = np.linalg.inv(np.full((3, 3), np.nan)); print("inv(NaN matrix): no exception, finite:", bool(np.isfinite(r).all()))
except Exception as e:
    print("inv(NaN matrix) raises", repr(e))
A = np.random.default_rng(0).normal(size=(30, 5)); A[7, 2] = np.nan
print("a single NaN row in SessionEM's history: column mean finite?", bool(np.isfinite(A.mean(0)).all()), "(NaN stays for every later window)")

# --- 4. eye traces on degenerate windows (extras.EyeNet.traces without the nets) ---
ex = importlib.import_module("tree.extras")
e = ex.EyeNet.__new__(ex.EyeNet)
ix = {c: i for i, c in enumerate(ex.CH_NAMES)}
e.idx = {k: [ix[c] for c in cs] for k, cs in ex.EYE_SETS.items()}
for name, x in (("zeros", np.zeros((41, 480), np.float32)), ("constant", np.ones((41, 480), np.float32)),
                ("3e38", np.full((41, 480), 3e38, np.float32) * np.sign(np.random.default_rng(1).normal(size=(41, 480))).astype(np.float32)),
                ("1e-30", (1e-30 * np.random.default_rng(2).normal(size=(41, 480))).astype(np.float32))):
    e.n, e.s1, e.s2 = 0, 0.0, 0.0
    with np.errstate(all="ignore"):
        T = e.traces(x)
    print(f"eye traces, first window {name}: finite {bool(np.isfinite(T).all())}, max |T| {float(np.abs(T).max()):.3g}")
e.n, e.s1, e.s2 = 0, 0.0, 0.0
rng = np.random.default_rng(3)
with np.errstate(all="ignore"):
    e.traces((20 * rng.normal(size=(41, 480))).astype(np.float32))
    T = e.traces(np.full((41, 480), 3e38, np.float32) * np.sign(rng.normal(size=(41, 480))).astype(np.float32))
    T2 = e.traces((20 * rng.normal(size=(41, 480))).astype(np.float32))
print("eye traces after one 3e38 window: this window finite", bool(np.isfinite(T).all()), "; next normal window finite", bool(np.isfinite(T2).all()),
      "max |T| of the next normal window", float(np.abs(T2).max()))
try:
    e.traces(np.zeros((41, 20), np.float32))
except Exception as err:
    print("eye traces on a 20-sample window ->", repr(err)[:120])
