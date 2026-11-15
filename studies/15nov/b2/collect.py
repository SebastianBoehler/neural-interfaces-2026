"""s284 collect: per-w, per-person lr10 (primary) vs w=0 (core1 reference, same code); mean, up-count. Usage: collect.py [DIR=runs/b2]"""
import glob, json, sys
D = sys.argv[1] if len(sys.argv) > 1 else "runs/b2"
R = {}
for w in ("0", "0.1", "0.3", "1"):
    R[w] = {j["person"]: j["ba"] for j in map(lambda f: json.load(open(f)), glob.glob(f"{D}/w{w}/fold*.json"))}
    print(w, len(R[w]), "folds")
for h in ("lr10", "lr3", "lr1"):
    b = R["0"]
    print(h, "w=0 mean", round(100 * sum(v[h] for v in b.values()) / max(len(b), 1), 2), "(core1 reference lr10 69.29)")
    for w in ("0.1", "0.3", "1"):
        ps = sorted(set(b) & set(R[w]))
        if ps:
            dl = [100 * (R[w][p][h] - b[p][h]) for p in ps]
            print(f"  w={w} n={len(ps)} mean {100*sum(R[w][p][h] for p in ps)/len(ps):.2f} delta {sum(dl)/len(dl):+.2f} up {sum(x > 0 for x in dl)}/{len(ps)}")
