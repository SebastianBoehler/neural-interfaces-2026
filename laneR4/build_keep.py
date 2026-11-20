"""Lane R4: keep-adapting variants of the final stack (cap250 + hard + CBraMod): fine-tuning (FT) and the causal personalisation (CSP)
are no longer retired by the recognition latch `recB`. Only submission.py is rewritten, every other file is copied byte for byte.

usage: build_keep.py ARM OUT_ZIP [SRC_ZIP]   or   build_keep.py ARM OUT_DIR SRC_DIR (package directory, returning fold packages)
ARMS (evidence/wave_20nov/laneR4_keepadapt.md):
  R0z  all switches off (zero-change control: must equal the source package bit for bit)
  R1   FT stays active after recognition (stock objective, labels = v5.1b's routed output, as the stored relabel at recognition)
  R2   R1 + CSP stays active (its history and P are not cleared at recognition)
  R3   R2 with the lite5 TTT objective (laneT/ttt_ft.py.inc, form lite5)
  R4   R3 + personal head in the FT pseudo-label: label = (1 - lam r) out_v5 + lam r h, lam 0.5, r the routed ramp, h the routed person's head
  R2s / R2q  R2 with post-recognition refits only every 2nd / 4th grid point (fewer refits, runtime fit)
Every anchor must occur exactly once (silent no-ops fail loudly)."""
import hashlib
import sys
from copy import copy
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ARMS = {  # keep_ft, keep_csp, ttt form, head lambda, late stride
    "R0z": (0, 0, None, 0.0, 1), "R1": (1, 0, None, 0.0, 1), "R2": (1, 1, None, 0.0, 1), "R3": (1, 1, "lite5", 0.0, 1),
    "R4": (1, 1, "lite5", 0.5, 1), "R2s": (1, 1, None, 0.0, 2), "R2q": (1, 1, None, 0.0, 4)}


def once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:70])
    return s.replace(old, new)


def patch(src: bytes, arm: str) -> bytes:
    keep_ft, keep_csp, form, lam, stride = ARMS[arm]
    s = src.decode("utf-8")
    if form:
        sys.path.insert(0, str(HERE.parent / "laneT"))
        import build_ttt
        s = build_ttt.patch(src, form).decode("utf-8")
    s = once(s, "class FineTune:", f"KEEP_FT, KEEP_CSP, HEAD_LAMBDA, LATE_STRIDE = {keep_ft}, {keep_csp}, {lam}, {stride}  # lane R4\n\n\nclass FineTune:")
    s = once(s, "self.refits, self.steps, self.dtheta = 0, 0, 0.0", "self.refits, self.steps, self.dtheta = 0, 0, 0.0\n        self.late = False  # lane R4: set by the ensemble once the session is recognised")
    s = once(s, "if (t - self.t0) % self.every == 0:", "if (t - self.t0) % self.every == 0 and (not self.late or ((t - self.t0) // self.every) % LATE_STRIDE == 0):")
    s = once(s, "self.recB if self.routed else not self.active).argmax()", "False if KEEP_CSP else (self.recB if self.routed else not self.active)).argmax()")
    s = once(s, "        self.last = dict(contrast=contrast, identity=identity, stage=stage)",
             "        self.head = (h, r) if r > 0 else None  # lane R4: the routed person's head posterior of this window\n"
             "        self.last = dict(contrast=contrast, identity=identity, stage=stage)")
    s = once(s, "    def step62(self, x):", '''    def head_label(self, out_v5):
        """lane R4: FT pseudo-label with the routed person's head posterior in it (HEAD_LAMBDA 0: v5.1b's output)"""
        head = self.route.head
        if not HEAD_LAMBDA or head is None:
            return out_v5
        lam = HEAD_LAMBDA * head[1]
        o = (1 - lam) * np.asarray(out_v5, dtype=np.float64) + lam * np.asarray(head[0], dtype=np.float64)
        return o / o.sum()

    def step62(self, x):''')
    s = once(s, "                label = out_v5\n            p_ft = self.ft.step(dict(nets), self.nets.last_inputs, label, self.relabel_prefix if self.r15 else None) \\\n                if (self.routed == 2 or not self.recB) else {}",
             "                label = self.head_label(out_v5)\n            self.ft.late = bool(self.recB)\n"
             "            p_ft = self.ft.step(dict(nets), self.nets.last_inputs, label, self.relabel_prefix if self.r15 else None) \\\n                if (KEEP_FT or self.routed == 2 or not self.recB) else {}")
    compile(s, "submission.py", "exec")
    return s.encode("utf-8")


if __name__ == "__main__":
    arm, out, src = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else HERE.parent / "runs/sealed/final_zips/final_stack_hardcb.zip"
    assert arm in ARMS
    if out.suffix == ".zip":
        assert not out.exists()
        with ZipFile(src) as z, ZipFile(out, "x") as o:
            for e in z.infolist():
                b = z.read(e.filename)
                o.writestr(copy(e), patch(b, arm) if e.filename == "submission.py" else b)
    else:
        out.mkdir(parents=True)
        for f in src.iterdir():
            (out / f.name).write_bytes(patch(f.read_bytes(), arm) if f.name == "submission.py" else f.read_bytes())
    print("built", arm, out, hashlib.sha256(out.read_bytes()).hexdigest() if out.is_file() else "")
