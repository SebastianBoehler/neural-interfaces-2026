import sys, time, copy, warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(1)
PKG = "/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026/runs/sealed/cand_v52b_pnets/"
x1 = torch.randn(1, 41, 480); x32 = torch.randn(32, 41, 480)
def times(m, x, n=6):
    out = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter(); m(x); out.append(round(1e3 * (time.perf_counter() - t0), 1))
    return out
for f in ("p_sbn_k2_s0_P01.pt", "p_sbn_sincshallow_ea_ptm7r_rot05_k20_s0_P01.pt"):
    m = torch.jit.load(PKG + f, map_location="cpu").eval()
    t0 = time.perf_counter(); c1 = copy.deepcopy(m); dc = 1e3 * (time.perf_counter() - t0)
    print(f, "deepcopy ms %.1f" % dc)
    print("   copy 1, batch 1, ms per call:", times(c1, x1))
    print("   copy 1, batch 32:", times(c1, x32, 4), "then batch 1 again:", times(c1, x1, 3))
    c2 = copy.deepcopy(m)
    print("   copy 2 (fresh deepcopy of the pristine), batch 1:", times(c2, x1, 4))
    c3 = copy.deepcopy(m)
    print("   copy 3, batch 32 first:", times(c3, x32, 3), "then batch 1:", times(c3, x1, 3))
