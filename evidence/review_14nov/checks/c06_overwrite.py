"""shufcombo's output name omits OCU_ORACLE / OCU_SYN (and OCU_DIR, BASES, RS_DIR ...). Oracle and synthetic runs of s188 used
the regular run's settings otherwise, so they wrote the regular file's name. Is the local regular file still a regular result?
Compares local runs/ with the Leonardo copy runs/leo_npz (independent machine) and prints file times."""
import glob, os, time
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")


def mb(f, arm):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files}); return np.mean([z[f"{bn}|{arm}"] for bn in b], 0), len(b)


for o in ("within0", "within1", "within2", "full0", "runs0"):
    n = f"shufcombo_{o}_ocu_gC5_all_ge_hb_ts_gp0_ttt18{o}L.npz"; a, b = f"{REPO}/runs/{n}", f"{REPO}/runs/leo_npz/{n}"
    (va, na), (vb, nb) = mb(a, "V6Fr"), mb(b, "V6Fr"); (pa, _), (pb, _) = mb(a, "P0r"), mb(b, "P0r")
    print(f"{o}: local mtime {time.strftime('%d %H:%M', time.gmtime(os.path.getmtime(a)))} UTC ({na} bases) | V6Fr - P0r local {np.mean(va - pa):+.2f} vs Leonardo {np.mean(vb - pb):+.2f} | "
          f"max |P0r diff| {np.abs(pa - pb).max():.2f}, max |V6Fr diff| {np.abs(va - vb).max():.2f}")
    logs = [(os.path.getmtime(f), os.path.basename(f)) for f in glob.glob(f"{REPO}/runs/*.log") if f"shufcombo_{o}_" not in f and os.path.getsize(f) < 2e5
            and any(s in open(f, errors="ignore").read(3000) for s in (f"SHUFCOMBO {o[:-1]} {o[-1]}: FT gC5 re-streamed ttt18{o}L,",))]
    for t, l in sorted(logs)[-6:]:
        txt = open(f"{REPO}/runs/{l}", errors="ignore").read(4000)
        flags = [s for s in ("OCU_ORACLE", "OCU_SYN", "OCU_TX", "SYN ", "OUTX", "GATE_P0", "labels (") if s in txt]
        print(f"      {time.strftime('%d %H:%M', time.gmtime(t))} {l}: {flags}")
