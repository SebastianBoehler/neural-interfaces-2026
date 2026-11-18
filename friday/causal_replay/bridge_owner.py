"""Run the frozen historical P0r producer with held labels inaccessible."""
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

BASE = Path("/mnt/lustre/work/gehler/gfh527")
ROOT = BASE / "ni26_s318_coral_fbl_20261117"
MEANS = BASE / "ni26_friday_causal_rebuild_20261118"
HERE = Path(__file__).resolve().parent
PROTO, ORDER, STACK = "all_loso", "full0", "B2_pool"
RAW = ROOT / "runs/chainout/s281_v66p_control_full0.npz"
GATE = ROOT / "runs/newmember/s281_v66p_control_full0_gate.npz"
PROVENANCE = ROOT / "runs/newmember/s281_v66p_control_full0_provenance.json"
PINS = {
    RAW: "4ae2d0d2654c99497f5ae303e23654361efa1cf09c19fafc6e53605573957250",
    GATE: "f28a174ab3fd0cc25cb55c991bc2618aae1505f61cbcd555a3d646b94f3359cb",
    PROVENANCE: "6353b8f45794d1c20ae7385be01bf57dc59b0029c6599578022d9c2171545ec4",
}
SLOTS = {
    1: ("fbcnet_ea_tf", "93048f398201e17db0ceb1814861319ba3590e7b986f98d15073740a064a047d",
        "904920d7e7e9bdbc30aa583a07179400673f07751eebbc8950bf30c8c1c1f2bd"),
    3: ("shallow_ea", "283e06cf00f98e3ebe074c262f90a179a28109d7506e25dadb7dbe67a9eb8939",
        "9cd2921638e7e6496a7b849f1bdfe131792a35d17bbc836c2bcfee0a448a1048"),
}


def sha(path):
    with path.open("rb") as fd:
        return hashlib.file_digest(fd, "sha256").hexdigest()


def pin(path, expected):
    if sha(path) != expected:
        raise ValueError(f"unexpected file identity {path}")


def stamp(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def blocked(key):
    return key.split("|")[0].lower() in {"y", "y_tr", "y_te", "label", "labels", "target", "targets"}


class LabelFreeArchive(Mapping):
    def __init__(self, raw, path):
        self.raw, self.path = raw, path
        self.files = [key for key in raw.files if not blocked(key)]

    def __getitem__(self, key):
        if blocked(key):
            raise RuntimeError(f"label access forbidden in {self.path}: {key}")
        return self.raw[key]

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __contains__(self, key):
        return key in self.files

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.raw.close()

    def close(self):
        self.raw.close()


@contextmanager
def guarded_load(pins=PINS, provenance_path=PROVENANCE):
    original = np.load
    opened = {}
    for path, expected in pins.items():
        pin(path, expected)
    owners = ("runs/", "data/", "studies/", "replay/", "experiments/")
    provenance = json.loads(provenance_path.read_text())["inputs_sha256"]
    expected = dict(pins)
    for location, digest in provenance.items():
        path = Path(location)
        matches = [i for i, part in enumerate(path.parts) if part + "/" in owners]
        if matches:
            expected[ROOT.joinpath(*path.parts[matches[0]:])] = digest

    def load(file, *args, **kwargs):
        path = Path(file).absolute()
        if "/runs/labels/" in str(path) or ("ttt18" in path.name and "_r15c250-" in path.name):
            raise RuntimeError(f"cached dependent output forbidden: {path}")
        if path not in opened:
            opened[path] = sha(path)
            if path in expected and opened[path] != expected[path]:
                raise ValueError(f"archive manifest mismatch {path}")
        value = original(file, *args, **kwargs)
        if hasattr(value, "files"):
            return LabelFreeArchive(value, path)
        raise ValueError(f"unexpected data format {path}")

    np.load = load
    try:
        yield opened
    finally:
        np.load = original


def replace_once(source, term, replacement):
    if source.count(term) != 1:
        raise ValueError("unexpected historical source structure")
    return source.replace(term, replacement)


def start_module(name, path, source):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def owner(order=ORDER):
    scheme, seed = order[:-1], order[-1]
    scope = dict(SCHEME=scheme, PSEED=seed, SCHEMES=scheme, PERMS=str(int(seed) + 1), STRICT="1", GATE_P0="1",
                 OCU_DIR=f"runs/ocushuf/{order}", FT_V="r15c250", REC_PFX="ttt7",
                 T6_DIR=str(ROOT / "runs/ttt/probs7"), T17_DIR=str(ROOT / "runs/diag/ttt/probs18"),
                 FT_TAG=f"ttt18{order}LPC", SAVE_OUT="in_memory_only", PROTO=PROTO, BASES=STACK)
    forbidden = ("LABEL_OUT", "LABEL_ARM", "HALF", "SYN_T", "OCU_SYN", "OCU_ORACLE", "OCU_TX",
                 "OUTX", "RUNOFF", "NUIS", "OCU_TS", "EM_LAM", "TS_LAM", "BTS", "TSEM_TGT",
                 "CS_NPZ", "CM_NPZ", "RS_DIR", "GE_NPZ", "HB_NPZ", "TS_NPZ", "TS_FILE", "NI26_REPLAY_DATA")
    if set(forbidden) & os.environ.keys():
        raise ValueError("historical diagnostic overrides are forbidden")
    os.environ.update(scope)
    sys.path[:0] = [str(ROOT / part) for part in ("replay", "experiments", "studies/13nov")]
    rlib = ROOT / "replay/rlib.py"
    start_module("rlib", rlib, replace_once(rlib.read_text(), '; Y = f["y"]', ""))
    shuffle = ROOT / "studies/13nov/shuffle.py"
    text = replace_once(shuffle.read_text(), 'len(A.L.Z[f"y|{p}"])', 'len(A.L.Z[f"idx|{p}"])')
    registration = next(line for line in text.splitlines() if line.startswith("DICTS = {"))
    text = replace_once(text, registration, "DICTS = (A.L.Z, A.pl.B, SC.B)")
    start_module("shuffle", shuffle, text)
    chain = ROOT / "studies/13nov/shufcombo.py"
    text, found, _ = chain.read_text().partition("\nif LABEL_OUT:  # FT pseudo-labels")
    if not found:
        raise ValueError("historical owner terminator changed")
    # P0r only needs frozen SessionBN predictions. Keep ft variables inaccessible.
    for name in ("CU6", "CU17"):
        line = next(line for line in text.splitlines() if line.startswith(name + " = "))
        text = replace_once(text, line, f"{name} = {{}}")
    text = replace_once(text, "return SC.person(out, p0)", "return out")
    return start_module("bridge_shufcombo", chain, text)


def corrected(inputs, order=ORDER):
    from member_inputs import ORDERS, pinned_delivery

    delivery = pinned_delivery(ROOT / "runs/shuffle_perms.npz")
    corrected = {}
    for slot, (member, file_hash, bank_hash) in SLOTS.items():
        path = MEANS / f"{member}_slot{slot}_means.npz"
        pin(path, file_hash)
        with np.load(path, allow_pickle=False) as z:
            if (tuple(z["orders"]) != ORDERS or str(z["member"]) != member
                    or str(z["bank_sha256"]) != bank_hash or not np.array_equal(z["delivery"], delivery)
                    or not np.array_equal(z["index"], np.arange(len(inputs)))):
                raise ValueError("causal mean provenance or layout differs")
            values = z["proba"][ORDERS.index(order)]
        if (values.shape != (len(inputs), 3) or not np.isfinite(values).all()
                or not np.allclose(values.sum(1), 1, rtol=0, atol=2e-6)):
            raise ValueError("causal slot probability bank invalid")
        corrected[slot] = values
    return corrected
