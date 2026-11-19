"""Assemble the measured stock-protected selector in one official Solver file."""
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cap250_fbl2_package"))
from build import CODE_HASH, SOLVER_MARKER, assemble

MEASURED_REV = "50a8451a51dcfb1c4a9ff5cde05580dd22cb3501"
MEASURED = (
    ("cap250c21e/fbl2.py", "736a1dbafce94058df803570ca49eb7cbdf55a2b03f5867305ab0c8e35474ea2",
     ("MEMBERS", "CAP_SHARE", "BOOST_SHARE", "FBL2Hook")),
    ("cap250c20e_shared/mirror.py", "f52e414494585fd2d6926e050dd2ac0462ebedc6c8f9981d16e19e6d6d02a3cb",
     ("Outer",)),
    ("cap250c21_gate/native/live_gate.py", "d03be2a74c621563e8ef8119197f775d5842135cf6816b7db8738c432a277dbf",
     ("LiveSelector",)),
)
RETURN = '        model.perm = channel_map(meta.get("ch_names"))\n        return model'


def measured_fragments(path, fingerprint, names, revision=MEASURED_REV):
    raw = subprocess.check_output(["git", "show", f"{revision}:friday/{path}"], cwd=HERE)
    if hashlib.sha256(raw).hexdigest() != fingerprint:
        raise ValueError(f"measured selector input changed: {path}")
    source = raw.decode("utf-8")
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            key = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
        else:
            continue
        if key in names:
            if key in found:
                raise ValueError(f"duplicate measured definition: {key}")
            found[key] = ast.get_source_segment(source, node)
    if found.keys() != set(names):
        raise ValueError(f"missing measured definitions: {path}")
    return [found[name] for name in names]


def selector_source():
    imports = "from copy import deepcopy\nfrom types import SimpleNamespace"
    fragments = [piece for item in MEASURED for piece in measured_fragments(*item)]
    return "\n\n\n".join([imports, *fragments])


def revised_with(raw, selector, selector_name):
    if hashlib.sha256(raw).hexdigest() != CODE_HASH:
        raise ValueError("unexpected frozen cap250 code")
    code = raw.decode("utf-8")
    if code.count(SOLVER_MARKER) != 1 or code.count(RETURN) != 1:
        raise ValueError("unexpected frozen Solver entry point")
    wrapper = HERE.joinpath("gate_model.py.inc").read_text().rstrip()
    if wrapper.count("LiveSelector(model)") != 1:
        raise ValueError("unexpected selector construction in the entry point")
    wrapper = wrapper.replace("LiveSelector(model)", f"{selector_name}(model)")
    body = selector + "\n\n\n" + wrapper
    code = code.replace(SOLVER_MARKER, body + "\n\n\n" + SOLVER_MARKER)
    code = code.replace(RETURN, RETURN.removesuffix("return model") + "return StockProtectedSolverModel(model)")
    compile(code, "submission.py", "exec")
    return code.encode("utf-8")


def revised(raw):
    return revised_with(raw, selector_source(), "LiveSelector")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(assemble(args.frozen, args.output, revised))
