"""Read the native raw FBCNet feature and derive matched first-visit heads."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from assets import digest, meta, sha
from inputs import BASE


MEMBER = 47
MODEL = "sbn_fbcnet_raw_k20_s0.pt"
EEG = BASE / "neural-interfaces-2026/data/proteus_windows.npz"
PROOF = {
    "P04": ("pilot1", "03427f00e576e1758d00086c8d60dbe5fb8bdf65dfcf1b2a16234f8377b88ede"),
    "P10": ("full1", "db43a2ef78c0c12a316eb3568ca5cff9162d12753b0cce427f05bae80feff7f3"),
}


def row_labels(path, rows):
    """Read only the requested int64 entries of the compressed labels member."""
    rows = np.asarray(rows, dtype=np.int64)
    assert rows.ndim == 1 and len(rows) and (np.diff(rows) > 0).all()
    with ZipFile(path) as archive, archive.open("y.npy") as data:
        version = np.lib.format.read_magic(data)
        assert version == (1, 0)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(data)
        assert shape == (2655,) and not fortran and dtype == np.dtype("<i8")
        assert rows[0] >= 0 and rows[-1] < shape[0]
        base = data.tell()
        parts = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
        labels = []
        for group in parts:
            data.seek(base + int(group[0])*dtype.itemsize)
            block = data.read(len(group)*dtype.itemsize)
            assert len(block) == len(group)*dtype.itemsize
            labels.append(np.frombuffer(block, dtype=dtype))
    result = np.concatenate(labels)
    assert np.isin(result, (0, 1, 2)).all()
    return result


def waveform_proof(who, package, people, visits):
    stage, expected = PROOF[who]
    path = BASE / f"ni26_js321_20261117_returning_other_{stage}/other/fbraw_{who}_s23_s0/proof.json"
    assert sha(path) == expected
    proof = meta(path)
    owner = people == f"Dreyer2026Proteus/{who}"
    held = np.flatnonzero(owner & np.isin(visits, ("2", "3")))
    train = np.flatnonzero(~(owner & np.isin(visits, ("2", "3"))))
    assert proof["fold"] == who+"_s23" and proof["family"] == "fbraw" and proof["seed"] == 0
    assert not proof["accuracy_computed"] and proof["status"] == "new_genuine_other_family_source_export_passed"
    assert (proof["train_count"], proof["held_count"]) == (len(train), len(held))
    assert proof["train_rows_sha256"] == digest(train) and proof["held_rows_sha256"] == digest(held)
    assert sha(package / MODEL) == proof["exports"]["20.0"]["sha256"]
    return dict(proof=expected, actor=sha(package / MODEL), train_rows=digest(train), held_rows=digest(held))


def live_waveform(model):
    found = [(stream, net) for stream, members in model.nets.groups for i, net in members if i == MEMBER]
    assert len(found) == 1 and model.ft.member["fbraw"] == MEMBER and model.ft.files["fbraw"] == MODEL
    stream, net = found[0]
    assert type(net).__name__ == "SharedFIRSJit" and not stream.align and not net.fblight
    assert net.m.original_name == "FBCNet"
    return stream, net


def forward_feature(net, x, cache):
    """Same SharedFIRSJit path with its actual final-layer input exposed."""
    if "sbn_bank" not in cache:
        cache["sbn_bank"] = net.m.spectral_filtering(x)
    state = net.m.spatial_conv(cache["sbn_bank"])
    score = net.m.temporal_layer(state.view(state.shape[0], state.shape[1], 4, 120))
    z = net.m.flatten_layer(score)
    assert z.ndim == 2 and z.shape[0] == 1 and torch.isfinite(z).all()
    p = torch.softmax(net.m.final_layer(z).double(), 1)
    return z, p


def unit(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    assert np.isfinite(v).all() and (norm > 1e-12).all()
    return v / norm


def build(features, labels, people, rows, who):
    owner = people == f"Dreyer2026Proteus/{who}"
    own, nonown = np.flatnonzero(owner), np.flatnonzero(~owner)
    assert (len(own), len(set(people[nonown]))) == ((189 if who == "P04" else 192), 9)
    mean = np.mean(features[nonown], axis=0, dtype=np.float64)
    scale = np.maximum(np.std(features[nonown], axis=0, dtype=np.float64), .001)
    values = unit((features-mean)/scale)
    rng = np.random.default_rng(16216)
    donors = sorted(set(people[nonown]))
    own_c, pop_c, pop_rows, counts = [], [], [], []
    for c in range(3):
        positive = own[labels[own] == c]
        n = len(positive)
        assert n > 40
        counts.append(n)
        quota, remainder = divmod(n, len(donors))
        source = []
        for i, p in enumerate(donors):
            ix = np.flatnonzero((people == p) & ~owner & (labels == c))
            assert len(ix) >= quota + (i < remainder)
            source.extend(rng.choice(ix, quota + (i < remainder), replace=False).tolist())
        assert len(set(source)) == n
        own_c.append(unit(values[positive].mean(0)))
        pop_c.append(unit(values[source].mean(0)))
        pop_rows.extend(source)
    own_c, pop_c = np.asarray(own_c), np.asarray(pop_c)
    assert own_c.shape == pop_c.shape == (3, features.shape[1])
    assert not np.array_equal(own_c, pop_c)
    assert (np.triu(np.abs(own_c@own_c.T - np.eye(3)), k=1) < .999).all()
    return dict(mean=mean, scale=scale, own=own_c, population=pop_c,
                own_rows=rows[own], population_rows=rows[np.asarray(pop_rows)], scale_rows=rows[nonown],
                class_counts=np.asarray(counts))
