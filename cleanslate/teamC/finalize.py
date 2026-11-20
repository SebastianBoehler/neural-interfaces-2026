"""Copy a team C package and bake a stream_eval variant's TTT settings into its weights.pt (the Solver reads only that).

python cleanslate/teamC/finalize.py --pkg DIR --variant A2 --out DIR2
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from stream_eval import VARIANTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    blob = torch.load(Path(a.pkg) / "weights.pt", map_location="cpu", weights_only=False)
    import submission as S
    unknown = set(VARIANTS[a.variant]) - set(S.DEFAULT_TTT)
    assert not unknown, unknown
    blob["ttt"] = {**blob["ttt"], **VARIANTS[a.variant]}
    torch.save(blob, out / "weights.pt")
    shutil.copy(Path(a.pkg) / "submission.py", out / "submission.py")
    info = dict(source=str(a.pkg), variant=a.variant, ttt=blob["ttt"],
                sha256={f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in ("submission.py", "weights.pt")})
    json.dump(info, open(out / "finalize.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
