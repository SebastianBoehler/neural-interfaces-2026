"""Lane R4b: dose arms between R1a and R4, and the confidence-gated keep-adapting form (evidence/wave_20nov/laneR4b_dose.md).
Built on laneR4/build_keep.py (imported, untouched): its R4 text is patched once more; every other package file is copied byte for byte.

usage: build_dose.py ARM OUT_ZIP SRC_ZIP        ARM from ARMS below, or a gated form  G<proxy>_<theta>  (proxy conf | agree | ent; theta a float, 'on' = keep always, 'off' = never)
Dose arms (all lite5 TTT objective after recognition, FT + head label as R4 unless stated):
  R4n   R4 without CSP after recognition      R4s2 / R4s4  post-recognition refit every 2nd / 4th grid point (20 / 40 windows)
  R4t2  2 optimiser steps per refit (R4: 5)   R4h1  head weight 1.0 in the pseudo-label (R4: 0.5)
Gated forms (R4 computed in full, its effect used only while the gate is open; gate closed = the final stack's own output):
  proxy conf   = mean over the last GATE_W post-recognition windows of the max of the system's own output posterior
  proxy ent    = mean over the same windows of 1 - H(out)/log 3 (prefix entropy: low = uncertain)
  proxy agree  = share of the last GATE_W post-recognition windows in which the routed person's head and the population mix choose the same class
  gate open (keep adapting: FT members in the mix, CSP fused) iff at least GATE_MIN post-recognition windows exist and proxy < theta. 'on' is bitwise R4, 'off' bitwise the final stack.
Every window's [recB, ramp, conf_out, conf_v5, agree, gate_open, 1 - normalised entropy of out] is kept in model.pxy_log (scoring reads it; never a label)."""
import hashlib
import sys
from copy import copy
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "laneR4"))
import build_keep as K  # noqa: E402
from build_keep import once  # noqa: E402

R4 = K.ARMS["R4"]
DOSE = {"R4n": (R4[0], 0) + R4[2:], "R4s2": R4[:4] + (2, None), "R4s4": R4[:4] + (4, None), "R4t2": R4, "R4h1": R4[:3] + (1.0,) + R4[4:]}
K.ARMS.update({k: v for k, v in DOSE.items()})
GATE_W, GATE_MIN = 40, 10


def patch(src: bytes, arm: str) -> bytes:
    gate = None
    if arm.startswith("G"):
        proxy, theta = arm[1:].split("_")
        assert proxy in ("conf", "agree", "ent"), arm
        gate = (proxy, None if theta == "on" else float("-inf") if theta == "off" else float(theta))
        base = "R4"
    else:
        base = arm
    assert base in K.ARMS and base not in ("R0z",), base
    s = K.patch(src, base).decode("utf-8")
    if arm == "R4t2":
        s = once(s, '"lite5": dict(steps=5,', '"lite5": dict(steps=2,')
    if gate is None:
        gp, gt, gw = "'none'", "None", GATE_W
    else:
        gp, gt, gw = repr(gate[0]), ("None" if gate[1] is None else ("float('-inf')" if gate[1] == float("-inf") else repr(gate[1]))), GATE_W
    s = once(s, "KEEP_FT, KEEP_CSP, HEAD_LAMBDA, LATE_STRIDE, LATE_AGES =",
             f"GATE_PROXY, GATE_THETA, GATE_W, GATE_MIN = {gp}, {gt}, {gw}, {GATE_MIN}  # lane R4b: None = gate always open\nKEEP_FT, KEEP_CSP, HEAD_LAMBDA, LATE_STRIDE, LATE_AGES =")
    s = once(s, "self.active, self.recB, self.v5hist, self.dead = True, False, [], False",
             "self.active, self.recB, self.v5hist, self.dead = True, False, [], False\n        self.pxy_log, self.gate_open = [], True  # lane R4b")
    s = once(s, "    def head_label(self, out_v5):", '''    def gate_decide(self):
        """lane R4b: is the keep-adapting gate open for this window (windows before it only)"""
        if GATE_THETA is None or not self.recB:
            return True
        rows = [r for r in self.pxy_log if r[0]]  # post-recognition windows so far
        if len(rows) < GATE_MIN:
            return False
        col = {"conf": 2, "agree": 4, "ent": 6}[GATE_PROXY]
        return bool(np.mean([r[col] for r in rows[-GATE_W:]]) < GATE_THETA)

    def cp_label(self, x, p):
        o = self.cf.step(x, p)
        L = self.cp.step(x, p, o, False if KEEP_CSP else (self.recB if self.routed else not self.active))
        return int((L if (not KEEP_CSP or not self.recB or self.gate_open) else o).argmax())

    def head_label(self, out_v5):''')
    s = once(s, "labels.append(int(self.cp.step(x, p, self.cf.step(x, p), False if KEEP_CSP else (self.recB if self.routed else not self.active)).argmax()))",
             "labels.append(self.cp_label(x, p))")
    s = once(s, "            self.ft.late = bool(self.recB)\n", "            self.ft.late = bool(self.recB)\n            self.gate_open = self.gate_decide()\n")
    s = once(s, "            mB = self.mix6(mix, dict(nets), p_ft)\n",
             "            mB = self.mix6(mix, dict(nets), p_ft if (not self.recB or self.gate_open) else {})\n")
    s = once(s, "        log[\"v6\"] += 1\n        log[\"ft\"] += int(bool(p_ft))\n",
             "        log[\"v6\"] += 1\n        log[\"ft\"] += int(bool(p_ft))\n"
             "        hd = getattr(self.route, \"head\", None)  # lane R4b proxies, never a label\n"
             "        self.pxy_log.append((float(self.recB), float(hd[1]) if hd is not None else 0.0, float(np.max(out) / np.sum(out)), float(np.max(out_v5) / np.sum(out_v5)),\n"
             "                             float(int(np.argmax(hd[0])) == int(np.argmax(mix))) if hd is not None else float('nan'), float(self.gate_open), 1.0 + float(np.sum((out / np.sum(out)) * np.log(np.clip(out / np.sum(out), 1e-12, 1)))) / float(np.log(3.0))))\n")
    compile(s, "submission.py", "exec")
    return s.encode("utf-8")


if __name__ == "__main__":
    arm, out, src = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    assert out.suffix == ".zip" and not out.exists()
    with ZipFile(src) as z, ZipFile(out, "x") as o:
        for e in z.infolist():
            b = z.read(e.filename)
            o.writestr(copy(e), patch(b, arm) if e.filename == "submission.py" else b)
    print("built", arm, out, hashlib.sha256(out.read_bytes()).hexdigest())
