"""Seal the CBraMod banks with Lane C's collector on arm C only; refuse any REVE receipt."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "laneC")]

import data  # noqa: E402

data.FULL_ARMS = "C"
import cbmod  # noqa: E402
import full_collect  # noqa: E402

assert full_collect.FULL_ARMS == "C"

if __name__ == "__main__":
    root, work = Path(sys.argv[1]), Path(sys.argv[2])
    for receipt in sorted(work.glob("reve_full_C_*.json")) + sorted(work.glob("external_C_?.json")):
        r = json.loads(receipt.read_text())
        assert r.get("backbone") == "cbramod" and r.get("checkpoint_sha") == cbmod.BLOB, f"not a CBraMod receipt: {receipt}"
    sys.argv = [sys.argv[0], str(root), str(work)]
    full_collect.main()
