import sys
sys.dont_write_bytecode = True
REPO = "/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026"
sys.path[:0] = [REPO + "/experiments", REPO + "/scripts"]
import torch, deep, session_bn as SB
for arch in ("fbcnet", "fblight", "sincshallow", "shallow"):
    net = deep.build(arch, 41, 480)
    n = SB.swap_bn(net, False, 20.0)
    named = list(net.named_parameters())
    cand = [(k, tuple(p.shape)) for k, p in named if p.dim() > 1 and p.shape[0] >= 3]
    sbn = {id(p) for mod in net.modules() if isinstance(mod, SB.SessionBN) for p in mod.parameters()}
    print(arch, "n_sbn", n, "n params", len(named), "| 'last' =", cand[-1], "| final names:", [(k, tuple(p.shape), p.requires_grad) for k, p in named[-3:]])
    print("    all params:", [(k, tuple(p.shape), p.requires_grad, id(p) in sbn) for k, p in named])
    print("    dropout modules:", [type(m).__name__ for m in net.modules() if "Dropout" in type(m).__name__], "| other norm:", [type(m).__name__ for m in net.modules() if "Norm" in type(m).__name__ and not isinstance(m, SB.SessionBN)])
