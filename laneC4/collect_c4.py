"""Seal the frozen-encoder banks with Lane C's collector on arm C only; refuse receipts of any other model or init."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "laneC")]

import data  # noqa: E402

data.FULL_ARMS = "C"
import full_collect  # noqa: E402

assert full_collect.FULL_ARMS == "C"

if __name__ == "__main__":
    root, work, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]  # tag: e.g. braincodes-pretrained
    receipts = sorted(work.glob("reve_full_C_*.json")) + sorted(work.glob("external_C_?.json"))
    assert len(receipts) == 3 + 36, f"expected 39 receipts, found {len(receipts)}"
    for receipt in receipts:
        r = json.loads(receipt.read_text())
        assert r.get("backbone") == tag and r.get("frozen_body") is True, f"not a {tag} receipt: {receipt}"
    sys.argv = [sys.argv[0], str(root), str(work)]
    full_collect.main()
