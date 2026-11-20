"""Lane P2: build the dispersion-corrected packages from a final-stack package that already carries lite5 (laneT/build_ttt.py lite5).

usage: build_p2.py FORM OUT SRC   FORM ds_l5 (lite5 + dispersion) | ds_all (lite5 + FBL2 + dispersion) | dsu_l5, dsu_all (same, correction only on unrecognised sessions); OUT .zip (SRC zip) or a directory (SRC dir:
returning fold packages). Only submission.py is rewritten (SharedFIRSJit.proba exports the segment contributions, DispEnsemble is the factory);
every other file is copied byte for byte. Nothing is switched off by default: the zero-change check mutates DISP_ON / FBL2_ON at runtime."""
import hashlib
import os
import sys
from copy import copy
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
OLD_RET = "        return torch.softmax(self.m.final_layer(self.m.flatten_layer(h)).double(), 1)\n"
NEW_RET = '''        f = self.m.flatten_layer(h)
        out = self.m.final_layer(f)
        if self.seg_on:  # lane P2: segment logit contributions of the head (4 x 3), by linearity: the head applied to f masked to one segment, minus its bias
            mask = (torch.arange(f.shape[1], device=f.device) % 4)[None, :] == torch.arange(4, device=f.device)[:, None]
            o = self.m.final_layer(torch.cat([f * mask, torch.zeros_like(f)])).detach().double().cpu().numpy()
            seg = o[:4] - o[4]
            self.seg = seg if np.abs(seg.sum(0) + o[4] - out.detach().double().cpu().numpy()[0]).max() < 1e-3 else None
        return torch.softmax(out.double(), 1)
'''
CLS = '    """SessionBN FBCNet / FBLightConvNet trace run from a filter-bank output shared within its input group."""\n'
FACTORY = 'model = StackEnsemble(z,'


def patch(src: bytes, form: str) -> bytes:
    assert form in ("ds_l5", "ds_all", "dsu_l5", "dsu_all")
    s = src.decode("utf-8")
    assert s.count(OLD_RET) == 1 and s.count(CLS) == 1 and s.count(FACTORY) == 1 and s.count("class Solver(CompetSolver):") == 1
    assert os.environ.get("BUILD_P2_ANY_FT") or s.count("TTTFineTune(z, self.w, dev, root)") == 1, "the source package must carry lite5 (build_ttt.py lite5 first)"
    inc = (HERE / "disp.py.inc").read_text().rstrip().replace("FBL2_ON = False", f"FBL2_ON = {form.endswith('all')}").replace("UNREC_ONLY = False", f"UNREC_ONLY = {form.startswith('dsu')}")
    s = s.replace(OLD_RET, NEW_RET).replace(CLS, CLS + "    seg_on, seg = False, None\n\n")
    s = s.replace("class Solver(CompetSolver):", inc + "\n\n\nclass Solver(CompetSolver):", 1).replace(FACTORY, "model = DispEnsemble(z,")
    compile(s, "submission.py", "exec")
    return s.encode("utf-8")


if __name__ == "__main__":
    form, out, src = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if out.suffix == ".zip":
        assert not out.exists()
        with ZipFile(src) as z, ZipFile(out, "x") as o:
            for e in z.infolist():
                b = z.read(e.filename)
                o.writestr(copy(e), patch(b, form) if e.filename == "submission.py" else b)
    else:
        out.mkdir(parents=True)
        for f in src.iterdir():
            (out / f.name).write_bytes(patch(f.read_bytes(), form) if f.name == "submission.py" else f.read_bytes())
    print("built", out, hashlib.sha256(out.read_bytes()).hexdigest() if out.is_file() else "")
