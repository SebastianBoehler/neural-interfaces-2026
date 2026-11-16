"""§281 shuffled chain through r15c250, with the inner gate held fixed.

Run control first: ADD=none SWAP=none NM_NAME=control python run_s_v66.py ORDER.
Then run each arm with its own NM_NAME; ZERO=1 checks the zero-weight arm.
Outputs are new s281_v66_* files and refuse to overwrite. This is NOT the full
v6.6 result: the count filter and arm-specific causal personalisation follow.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I omits the script directory
from s281_guard import verify, verify_archive

REPO = HERE.parents[2]


def replace_once(src, old, new):
    assert src.count(old) == 1, f"expected one source anchor: {old[:100]!r}"
    return src.replace(old, new, 1)


def patch_runner(src):
    replacements = (
        ('f"ttt18{O}L_fb_gC5*"', 'f"ttt18{O}LPC_fb_r15c250*"'),
        ('FT_V="gC5"', 'FT_V="r15c250"'),
        ('E.get("FT_TAG", f"ttt18{O}L")', 'f"ttt18{O}LPC"'),
        ('f"runs/chainout/nm_{NAME}_{O}.npz"', 'f"runs/chainout/s281_v66_{NAME}_{O}.npz"'),
        ('if p != p0 or (P == np.arange(N)).all():', 'if CURRENT["arm"] != "V6Fr" or p != p0 or (P == np.arange(N)).all():'),
        ('p = f"{REPO}/runs/newmember/{NAME}_{os.path.basename(p)}"',
         'p = f"{REPO}/runs/newmember/s281_v66_{NAME}_{os.path.basename(p)}"; assert not os.path.exists(p), p'),
        ('runpy.run_path(f"{REPO}/studies/13nov/shufcombo.py", run_name="__main__")', 'execute_chain()'),
    )
    for old, new in replacements:
        src = replace_once(src, old, new)
    return src


def patch_chain(src):
    src = replace_once(src, '    S.apply({p: (P if p == p0 else np.arange(S.N[p])) for p in S.PR})',
                       '    CURRENT["arm"] = arm\n    S.apply({p: (P if p == p0 else np.arange(S.N[p])) for p in S.PR})')
    src = replace_once(src, 'g = gm.gate(GB[bn, ref] if GATE_P0 and ref else b, p0)',
                       'g = capture_gate(bn, arm, gm.gate, GB[bn, ref] if GATE_P0 and ref else b, p0)')
    src = replace_once(src, 'np.savez_compressed(f"{REPO}/{SAVE_OUT}", **OUTS)', 'checked_save(OUTS)')
    return src


def main():
    assert len(sys.argv) == 2, "supply one delivery order"
    name, add, swap = (os.environ[k] for k in ("NM_NAME", "ADD", "SWAP"))
    assert name.isidentifier(), "NM_NAME must be a simple identifier"
    zero = os.environ.get("ZERO") == "1"
    is_control = add == swap == "none"
    assert (name == "control") == is_control, "only the unmodified arm may use NM_NAME=control"
    for key in ("T17_DIR", "FT_TAG", "FT_V", "GE_NPZ", "HB_NPZ", "CS_NPZ", "TS_NPZ"):
        assert key not in os.environ, f"{key} override forbidden"
    order = sys.argv[1]
    out = REPO / f"runs/chainout/s281_v66_{name}_{order}.npz"
    audit = REPO / f"runs/newmember/s281_v66_{name}_{order}_gate.npz"
    reference = REPO / f"runs/newmember/s281_v66_control_{order}_gate.npz"
    archive = REPO / f"runs/s281_v66_reference/s242c250c_{order}.npz"
    assert not out.exists() and not audit.exists(), "refusing to overwrite §281 output"
    assert is_control or reference.exists(), f"run control first: {reference}"
    assert not is_control or archive.is_file(), f"missing archived §257 control: {archive}"
    gates, current = {}, {"arm": None}
    context = {"__name__": "__main__", "__file__": str(HERE / "run_s.py"), "CURRENT": current}

    def capture_gate(bn, arm, gate, value, proto):
        result = gate(value, proto)
        if arm == "V6Fr":
            assert f"{bn}|input" not in gates, f"duplicate gate {bn}"
            gates[f"{bn}|input"], gates[f"{bn}|gate"] = np.array(value, copy=True), np.array(result, copy=True)
        return result

    def checked_save(outputs):
        v6 = {k: v for k, v in outputs.items() if k.endswith("|V6Fr")}
        assert len(v6) == 6, f"expected six V6Fr bases, got {list(v6)}"
        records = {**gates, **v6}
        if is_control:
            verify(records, records, zero=True)
            with np.load(archive) as z:
                verify_archive(outputs, z)
        else:
            with np.load(reference) as z:
                verify(dict(z), records, zero=zero)
        audit.parent.mkdir(parents=True, exist_ok=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(audit, **records)
        np.savez_compressed(out, **outputs)
        print("archived control matched; gate data saved" if is_control else "gate inputs and decisions verified; output saved", flush=True)

    def execute_chain():
        path = REPO / "studies/13nov/shufcombo.py"
        context.update(__file__=str(path), capture_gate=capture_gate, checked_save=checked_save)
        exec(compile(patch_chain(path.read_text()), str(path), "exec"), context)

    context["execute_chain"] = execute_chain
    exec(compile(patch_runner((HERE / "run_s.py").read_text()), str(HERE / "run_s_v66.py"), "exec"), context)


if __name__ == "__main__":
    main()
