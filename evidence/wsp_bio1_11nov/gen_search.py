"""Do the public Graz cue sequences come from the OpenViBE Lua stimulator (ordered list + K random swaps, C rand()
seeded with the start time)? Brute-force seeds for several generator variants and look for EXACT matches.
Labels only; no EEG."""
import sys, time, collections
import numpy as np
z = np.load("<repo>/data/proteus_windows.npz", allow_pickle=True)
y = z["y"]; per = np.array([s[-3:] for s in z["subject"]]); ses = z["session"].astype(str); task = z["task"].astype(str); run = z["run"]
runs = collections.OrderedDict()
for i in range(len(y)):
    if task[i].endswith("Graz"):
        runs.setdefault((per[i], ses[i], task[i], int(run[i])), []).append(int(y[i]))
P3 = lambda seq: sum(int(c) * 3 ** k for k, c in enumerate(seq))  # noqa: E731
targets = {n: {P3(v): k for k, v in runs.items() if len(v) == n} for n in (30, 36)}
print({n: len(t) for n, t in targets.items()}, "target runs", flush=True)
LO, HI = int(sys.argv[1]), int(sys.argv[2]); CH = 2_000_000
W_ = 3 ** np.arange(36, dtype=np.uint64)


def msvc(state):
    state = (state * np.uint32(214013) + np.uint32(2531011)).astype(np.uint32)
    return state, ((state >> np.uint32(16)) & np.uint32(0x7fff)).astype(np.int64)


def gen(seeds, n, K, formula, skip):
    st = seeds.astype(np.uint32); seq = np.tile(np.array([2, 1, 0] * (n // 3), np.int8), (len(seeds), 1)); rows = np.arange(len(seeds))
    for _ in range(skip):
        st, _r = msvc(st)
    for _ in range(K):
        st, r1 = msvc(st); st, r2 = msvc(st)
        if formula == 51:   # Lua 5.1: (rand() % RAND_MAX) / RAND_MAX
            a = ((r1 % 32767) * n) // 32767; b = ((r2 % 32767) * n) // 32767
        else:               # Lua 5.2+: rand() / (RAND_MAX + 1)
            a = (r1 * n) // 32768; b = (r2 * n) // 32768
        va, vb = seq[rows, a].copy(), seq[rows, b].copy(); seq[rows, a] = vb; seq[rows, b] = va
    return (seq.astype(np.uint64) * W_[:n]).sum(1)


found = []
variants = [(K, f, s) for f in (51, 52) for s in (0, 1) for K in ("n/2", "n/3", "n")]
for vi, (Kn, formula, skip) in enumerate(variants):
    t0 = time.time(); hits = 0
    for n in (30, 36):
        K = {"n/2": n // 2, "n/3": n // 3, "n": n}[Kn]; tk = np.array(list(targets[n]), np.uint64)
        for lo in range(LO, HI, CH):
            seeds = np.arange(lo, min(lo + CH, HI), dtype=np.int64); code = gen(seeds, n, K, formula, skip); m = np.isin(code, tk)
            for i in np.flatnonzero(m):
                found.append((Kn, formula, skip, n, int(seeds[i]), targets[n][int(code[i])])); hits += 1
    print(f"variant K={Kn} lua{formula} skip{skip}: {hits} exact matches in seeds [{LO}, {HI}) {time.time() - t0:.0f}s", flush=True)
    if hits:
        for h in sorted(found, key=lambda x: x[4])[-min(hits, 50):]:
            print("   ", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(h[4])), h[4], h[5])
