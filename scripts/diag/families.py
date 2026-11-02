"""Family arguments for eval_external_teacher.py from the checkpoints present (overnight 2 Nov).

Usage: python scripts/diag/families.py TAG [TAG ...] [--with PREFIX+PREFIX]
Prints ``NAME=prefix+...`` for every ARCH_ea-TAG with seeds 0-5 on disk (one family per
architecture), ``all_TAG`` (all of them) and, with --with, each family and all_TAG
combined with the given prefixes (e.g. the packaged teacher).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
args, extra = sys.argv[1:], ""
if "--with" in args:
    extra = args[args.index("--with") + 1]
    args = args[:args.index("--with")]
out = []
for tag in args:
    fams = sorted({p.name.rsplit("-s", 1)[0] for p in (ROOT / "runs/pretrain").glob(f"*_ea-{tag}-s0.pt")})
    fams = [f for f in fams if all((ROOT / "runs/pretrain" / f"{f}-s{s}.pt").exists() for s in range(6))]
    own = [f"{f.split('_')[0]}_{tag}={f}" for f in fams]
    if len(fams) > 1:
        own.append(f"all_{tag}=" + "+".join(fams))
    out += own
    if extra:
        out += [f"{o.split('=')[0]}+x={o.split('=')[1]}+{extra}" for o in own]
print(" ".join(dict.fromkeys(out)))
