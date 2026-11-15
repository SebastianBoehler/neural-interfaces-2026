"""s263 collect: python collect.py DIR [VARIANT ...]. Z is checked first (within 0.1 point per person of core1's
stored all_loso lr10); variants A-D are printed only if Z passes. Fails loudly on a missing fold or a key mismatch."""
import json, sys, os
import numpy as np
D = sys.argv[1]; VS = sys.argv[2:] or ["A", "B", "C", "D"]
REF = json.load(open(os.path.dirname(__file__) + "/../../../runs/diag/ts_core/res/car_ov17_fs2_t0.1_le_k2.json"))["heads"]["lr10"]["all_loso"]

def folds(v):
    fs = [json.load(open(f"{D}/{v}/fold{i}.json")) for i in range(10)]
    assert len({f["key"] for f in fs}) == 1, f"{v}: mixed cache keys"
    return {f["person"]: f for f in fs}

z = folds("Z"); worst = max(abs(z[p]["ba"]["lr10"] - REF[p]) for p in REF)
print(f"Z vs core1: max |diff| {100 * worst:.3f} points -> {'PASS' if worst < 1e-3 else 'FAIL'}")
if worst >= 1e-3: sys.exit(1)
ref = np.array([REF[p] for p in sorted(REF)])
for v in VS:
    f = folds(v); b = np.array([f[p]["ba"]["lr10"] for p in sorted(REF)])
    print(f"{v}: lr10 {100 * b.mean():.2f} (core1 {100 * ref.mean():.2f}) delta {100 * (b - ref).mean():+.2f}, up {(b > ref).sum()}/10,",
          " ".join(f"{100 * x:+.1f}" for x in (b - ref)), f"| lr1 {100 * np.mean([f[p]['ba']['lr1'] for p in f]):.2f} lr3 {100 * np.mean([f[p]['ba']['lr3'] for p in f]):.2f}")
