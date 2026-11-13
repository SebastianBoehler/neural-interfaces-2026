"""RESTSHUF (ledger s176): the rest-class shallow net (studies/11nov/rest4/rest4.py, arch sh, cond r1) with its test
sessions streamed in a delivery order through deliver.py (causal alignment and SessionBN state follow the order).
Saved to runs/diag/restshuf/<SCHEME><PSEED>/rest4_sh_r1-<proto>-s<seed>.npz, keyed by window index.
Usage (Leonardo): SCHEME=within PSEED=0 python studies/13nov/restshuf.py SEED | smoke"""
import importlib.util, os, sys
from pathlib import Path
import numpy as np
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); SCHEME = os.environ["SCHEME"]; PSEED = int(os.environ["PSEED"])
_spec = importlib.util.spec_from_file_location("rest4", WS / "studies/11nov/rest4/rest4.py"); R4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R4)
sys.path.insert(0, str(WS / "studies/13nov"))
import deliver  # noqa: E402
PERMS = np.load(WS / "runs/shuffle_perms.npz")


def main():
    smoke = sys.argv[1] == "smoke"; seed = 0 if smoke else int(sys.argv[1]); out = WS / "runs/diag/restshuf" / f"{SCHEME}{PSEED}" / ("smoke" if smoke else "")
    out.mkdir(parents=True, exist_ok=True); d, d2, sel = R4.data()
    for proto in (("seen_session",) if smoke else ("all_loso", "seen_session")):
        deliver.install(R4.protocol, deliver.rank_from(PERMS, SCHEME, PSEED, proto))
        try:
            R4.run("sh", "r1", d, d2, sel, proto, seed, out, epochs=1 if smoke else 0)
        finally:
            deliver.uninstall(R4.protocol)
    if smoke:
        print("SMOKE PASS", flush=True)


if __name__ == "__main__":
    main()
