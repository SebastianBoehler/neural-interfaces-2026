"""Check whether a different MANAS personal head can affect the pinned visit-3 fold streams."""

import argparse
import ast
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cleanslate/harness"))
from data import PEOPLE, RETURNING, plans

ZIP_SHA = "42fb63cf1ae9f54cf48315b00894b3788bc65400e75e06bd208ffe3753c43b07"
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
ORDERS = range(71800, 71816)


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def signed_gate(source):
    (body,) = [node.args[0].value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "compile"
               and len(node.args) == 3 and isinstance(node.args[1], ast.Constant)
               and node.args[1].value == "<MANAS A1>"]
    code = ast.parse(body)
    funcs = [n for n in code.body if isinstance(n, ast.FunctionDef)
             and n.name in ("center", "covariance", "fingerprint")]
    (decoder,) = [n for n in code.body if isinstance(n, ast.ClassDef) and n.name == "SessionDecoder"]
    (identify,) = [n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == "_identify"]
    ns = {"np": np}
    exec(compile(ast.Module(body=[*funcs, identify], type_ignores=[]), "<signed MANAS A1 gate>", "exec"), ns)
    if len(funcs) != 3 or 'if self.recipe == "A1" and self.person is not None:' not in body:
        raise ValueError("unexpected signed MANAS personal-head path")
    return ns


def run(a):
    if digest(a.zip) != ZIP_SHA or digest(a.data) != DATA_SHA:
        raise ValueError("unexpected quarter ZIP or PROTEUS export digest")
    with TemporaryDirectory() as tmp, ZipFile(a.zip) as archive:
        gate = signed_gate(archive.read("submission.py").decode())
        weights = Path(tmp) / "team_a.pt"
        with archive.open("team_a.pt") as source, weights.open("wb") as out:
            shutil.copyfileobj(source, out)
        bank = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        if bank["recipe"] != "A1" or bank["arch"] != "manas":
            raise ValueError("unexpected MANAS package")
        names = tuple(bank["names"])
        keys = ("gallery", "identity_scale", "identity_radius", "enrolled")
        bank = {k: np.asarray(bank["bank"][k]) for k in keys}
    with np.load(a.data, allow_pickle=False) as z:
        person = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]])
        visits = z["session"]
        channels = tuple(str(x).removeprefix("EEG ").split("-", 1)[0] for x in z["ch_names"])
        raw = gate["center"](z["X"][:, [channels.index(c) for c in names]])
    if len(names) != 41 or sorted(set(person)) != PEOPLE or not bank["enrolled"].all():
        raise ValueError("unexpected montage or enrollment")
    cov = gate["covariance"](raw)
    fingerprints = gate["fingerprint"](raw, cov)
    gallery = np.stack([fingerprints[(person == p) & (visits == "1")].mean(0) for p in PEOPLE])
    np.testing.assert_allclose(bank["gallery"], gallery, atol=1e-6, rtol=0)
    scale = 1 / np.maximum(gallery.std(0), 0.08)
    distances = np.sqrt(np.mean(((gallery[None] - gallery[:, None]) * scale) ** 2, axis=-1))
    radius = np.quantile(distances[~np.eye(len(gallery), dtype=bool)], 0.10) * 0.8
    np.testing.assert_allclose(bank["identity_scale"], scale, atol=0, rtol=2e-6)
    np.testing.assert_allclose(bank["identity_radius"], radius, atol=1e-6, rtol=0)
    data = {"stream": np.char.add(np.char.add(person, "|"), visits)}
    streams = [f"{who}|3" for who in RETURNING]
    records = []
    for order in ORDERS:
        phase = "fit" if order < 71808 else "held"
        for stream, rows in plans(data, order, streams).items():
            who = stream.split("|")[0]
            with np.load(a.diet_tapes / phase / f"ret_{who}-{order - order % 4}.npz", allow_pickle=False) as t:
                np.testing.assert_array_equal(rows, t[f"rows_{order}_3"])
            state = SimpleNamespace(bank=bank, identity_sum=np.zeros(gallery.shape[-1], np.float64),
                                    person=None, streak=None, recipe="A1")
            for state.age, row in enumerate(rows, 1):
                gate["_identify"](state, raw[row], cov[row])
                if state.person is not None:
                    break
            records.append(dict(order=order, person=who, phase=phase, windows=len(rows),
                                activation=None if state.person is None else
                                dict(delivery=state.age, person=PEOPLE[state.person])))
    result = dict(zip_sha256=ZIP_SHA, data_sha256=DATA_SHA, records=records,
                  gallery_max_abs=float(np.max(np.abs(bank["gallery"] - gallery))),
                  scale_max_rel=float(np.max(np.abs(bank["identity_scale"] / scale - 1))))
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    for phase in ("fit", "held"):
        block = [r for r in records if r["phase"] == phase]
        active = [r for r in block if r["activation"] is not None]
        print(f"{phase}: {len(active)}/{len(block)} activated, {sum(r['windows'] for r in block)} pinned windows")
        for item in active:
            print(item)
    print("Only the signed EEG identity gate ran. No task labels or classifier scores were opened.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("zip", "data", "diet_tapes", "out"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
