"""Release F enrolled ZIP derivation and exact NI-V parity checks (inside Slurm jobs)."""
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def enrol():
    root = Path("zips/enrolled")
    root.mkdir()
    with ZipFile("zips/hss_n25.zip") as old, ZipFile("zips/hss_n25_fc.zip") as new:
        source = old.read("submission.py")
        fused = new.read("submission.py")
        assert old.namelist() == new.namelist()
        assert [k for k in old.namelist() if old.read(k) != new.read(k)] == ["submission.py"]
    for k in range(9):
        for tag in ("hss", "hss_n25"):
            name = f"{tag}_k{k}.zip"
            (root / name).symlink_to(Path("../../../ni26_niv_finalA_20261121/zips/enrolled") / name)
        with ZipFile(root / f"hss_n25_k{k}.zip") as old:
            assert old.read("submission.py") == source
            with ZipFile(root / f"hss_n25_fc_k{k}.zip", "x") as new:
                for entry in old.infolist():
                    data = fused if entry.filename == "submission.py" else old.read(entry.filename)
                    new.writestr(entry, data)


def compare(left, right, select=None):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        keys = {k for k in a.files if k.endswith("|rows") or k.endswith("|labels")}
        if select is None:
            assert keys == {k for k in b.files if k.endswith("|rows") or k.endswith("|labels")}, left
        pairs = [(k, k.replace("full|s71024|", "full|0|", 1)) if select else (k, k)
                 for k in keys if select is None or k.startswith(select)]
        assert pairs, left
        count = 0
        for x, y in pairs:
            if not np.array_equal(a[x], b[y]):
                raise ValueError(f"NI-V identity failed: {left} vs {right}: {x}")
            if x.endswith("|labels"):
                count += a[x].size
        return count


def smoke():
    out = Path("out")
    parent = out / "hss_n25/smoke.npz"
    assert compare(parent, out / "hss_n25_fc/smoke.npz") > 0
    assert compare(parent, Path("../ni26_niv_finalE_20261121/out/hss_n25/p5_c0.npz"), "full|s71024|") > 0
    print("OK: fresh fused/parent and pinned release E rows and labels")


def collect():
    out = Path("out")
    counts = {"B-new": 0, "B-ret": 0}
    for person in range(9):
        for chunk in range(4):
            name = f"p{person}_c{chunk}.npz"
            counts["B-new"] += compare(out / "hss_n25" / name, out / "hss_n25_fc" / name)
            assert (out / "hss" / name).is_file(), name
        name = f"p{person}_c0.npz"
        counts["B-ret"] += compare(out / f"ret_hss_n25_k{person}" / name,
                                   out / f"ret_hss_n25_fc_k{person}" / name)
        assert (out / f"ret_hss_k{person}" / name).is_file(), name
    assert all(counts.values()), counts
    report = out / "report_hss_n25_fc_vs_hss.json"
    results = json.loads(report.read_text())
    results["headline"] = "full-session shuffle (fresh orders 71032–71039)"
    results["parity_vs_nine25"] = {"labels_compared": counts, "changed": 0}
    results["sealed_weighted"]["assumption"] = "57/43 proxy; sealed cohort weights are unverified"
    report.write_text(json.dumps(results, indent=1) + "\n")
    print(results["parity_vs_nine25"])


if __name__ == "__main__":
    {"enrol": enrol, "smoke": smoke, "collect": collect}[sys.argv[1]]()
