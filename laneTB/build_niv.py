"""Lane TB-verify NI-V: build hardcbB_TB25 / hardcbB_TB0 (Team B nine members) from the NI-V hardcbB.zip. Run as a batch step on Leonardo.
python build_niv.py HARDCBB.zip BANK.pt OUT_DIR   (repo layout under ROOT: submissions/final_stack/team_b, patch_dead.py, cleanslate/teamB/submission.py)"""
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

R = Path(sys.argv[0]).resolve().parent
sys.path.insert(0, str(R / "submissions/final_stack/team_b"))
import build  # noqa: E402

src, bank_path, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
pack = torch.load(bank_path, map_location="cpu", weights_only=True)
print("pack keys", sorted(pack), "people", pack.get("people"), "first_only", pack.get("first_only"), "ext", repr(pack.get("ext")), flush=True)
assert list(pack["people"]) == [f"P{i:02d}" for i in range(1, 11)] and not pack.get("first_only") and pack.get("ext", "") == "", "bank is not the Scherer-free all-PROTEUS pack"
blob = build.bank(bank_path, 3)
for tag, w in (("TB25", 0.25), ("TB0", 0.0)):
    with ZipFile(src) as zi:
        dest = out / f"hardcbB_{tag}.zip"
        with ZipFile(dest, "x", ZIP_DEFLATED) as zo:
            for e in zi.infolist():
                data = zi.read(e.filename)
                if e.filename == "submission.py":
                    old = data.decode()
                    data = build.revised(old, w, 3, blob).encode()
                    assert f"TEAM_B_MIX = {w!r}" in data.decode() and "TeamBlendStack(z," in data.decode()
                zo.writestr(e, data)
            zo.writestr("team_b.pt", blob)
    with ZipFile(dest) as c:
        assert c.testzip() is None
    print("built", dest, dest.stat().st_size, flush=True)
