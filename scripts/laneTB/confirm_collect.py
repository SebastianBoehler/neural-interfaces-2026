"""Collect the final Team B confirmation (hard base): paired nine15/nine25 minus hard on the new orders, alone and pooled
with the 8 Lane TB-verify orders, people then orders resampled (verify_collect.contrast).

python confirm_collect.py CONFIRM_OUT VERIFY_OUT --data NPZ --orders 71156,...,71171 --verify-orders 71124,...,71131 --json OUT
"""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_collect as V  # noqa: E402

ARMS = V.ARMS


def pooled_dir(parts, tmp):
    """One directory with every pred file of the given roots (names carry the first order, so they do not collide)."""
    root = Path(tmp)
    for src in parts:
        for f in src.glob("*/*/pred_*.npz"):
            link = root / f.relative_to(src)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(f)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("confirm", type=Path)
    p.add_argument("verify", type=Path)
    p.add_argument("--data", required=True)
    p.add_argument("--orders", required=True)
    p.add_argument("--verify-orders", required=True)
    p.add_argument("--json", type=Path, required=True)
    a = p.parse_args()
    data = V.D.load(a.data)
    new = [int(s) for s in a.orders.split(",")]
    old = [int(s) for s in a.verify_orders.split(",") if s]
    sets = {"new": (new, [a.confirm / "hard"])}
    if old:  # an empty --verify-orders scores the new orders alone (collector self-check on Lane TB-verify output)
        sets["pooled"] = (sorted(new + old), [a.confirm / "hard", a.verify / "hard"])
    result, rows = {}, []
    for name, (orders, roots) in sets.items():
        with TemporaryDirectory() as parts, TemporaryDirectory() as merged:
            V.merge(pooled_dir(roots, parts), Path(merged), orders)
            cells = V.load(Path(merged), orders, data)
        for arm in ARMS[1:]:
            key = f"hard_{arm}_{name}"
            result[key] = V.contrast(cells[arm], cells["base"], orders)
            rows.append(V.line(f"hard {arm} − hard ({len(orders)} orders: {name})", result[key]))
    a.json.write_text(json.dumps(dict(new_orders=new, verify_orders=old, contrasts=result), indent=1) + "\n")
    print("| Contrast | Held-new Δ [80%] | Returning Δ [80%] | P04 / P10 | J | 57/43* | held-new per order |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
