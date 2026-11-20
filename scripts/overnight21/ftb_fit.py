"""Lane FT-B6s stage B: one Sinc SessionBN family fit for one fold scope, with a chosen pretrained trunk (m7rx / m7rc / m7rz).
usage: ftb_fit.py TAG KIND HELD SEED OUT   scopes from production hardcb/jobs.tsv (new: 9 other people; ret: all ten, first-session-only the returning person).
Same recipe as cleanslate/cap250/families/sbn_sinc.py (cfg sincshallow_ea_ptm<TAG>_rot05_sbn, trace prefix and package index 53+seed unchanged);
only the trunk file differs. Runs inside a Slurm GPU job through cleanslate/cap250/slurm.sh."""
import os
import sys
from pathlib import Path

tag, kind, held, seed, out = sys.argv[1:6]
REL = Path(os.environ["CAP250_RELEASE"])
sys.path.insert(0, str(REL / "cleanslate/cap250/families"))
jobs = [line.split("\t") for line in (Path(os.environ["FTB_JOBS"]).read_text().splitlines())]
row = [r for r in jobs if r[0] == kind and r[1] == held]
assert len(row) == 1, (kind, held)
people, first = row[0][2], ("" if row[0][3] == "-" else row[0][3])
sys.argv = ["sbn_sinc.py", "train", "--people", people, *(["--first-session-only", first] if first else []),
            "--data", os.environ["CAP250_DATA"], "--seed", seed, "--out", out]
from _sbn import run  # noqa: E402

run(__doc__, f"sincshallow_ea_pt{tag}_rot05_sbn", ((53, "sbn_sincshallow_ea_ptm7r_rot05_k20", 20),))
