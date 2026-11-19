"""Read only outcome-free source and independently signed native parent inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from fixture import DATA_SHA, SEEDS, schedule

PARENT = Path("/parent")
NOMINAL = Path("/external.npz")
LOCAL = Path("/local.npz")
LOCAL_SHA = "3a4b41c4ad5d4a7634915c17f76473e78d6bec4641a266bbbaeaa45d84638a3c"
PARENT_SEAL = "4c761f40e852d6d03e7a86f00f99324d6546f149d73411299099c3a31c36ad0f"
TRIAL = "6a569180d343c038558fe7eccb7903a36e2bdf694af0df3687bff3b17049fbcc"
STOCK = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
ACTOR_LOCK = "0292022abd39067cefebd0e3a0bd8f1834a04bfc67d7d41081cfe6abaae8ebd2"
ACTORS = (
    "ec33684850e902479f3bf28e472a9ea590e64fa759b0b03704daf7aff1ea4018",
    "ed83b6b5b2a2765f82afa4a2fcb3f2c526d24a32e8b789b527268bde67b5401e",
)
F32_SMOKE = "d01c559bce326a7400fb983a06565aeae4acedd7d86502c6126da93c42370f22"
SITES = "P2 Pz CPz Cz P1 P3 P5 CP3 C3 P6 P4 CP4 C4 FC4 F4 F3 Fz FC3 FCz".split()
ARMS = ("stock_A", "stock_B", "hard", "zero", "lag_sham", "current")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pin(path, digest):
    if sha(path) != digest:
        raise ValueError(f"immutable input changed: {path}")


def stamp(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    return sha(path)


def scope():
    pin(NOMINAL, DATA_SHA)
    with np.load(NOMINAL, allow_pickle=False) as source:
        subject, session, interface, names, onset = (
            source[key] for key in ("subject", "session", "interface", "ch_names", "onset")
        )
    return subject, session, interface, names, onset, schedule(subject, session)


def input_waves(subject, session, interface, names, onset):
    pin(LOCAL, LOCAL_SHA)
    with np.load(NOMINAL, allow_pickle=False) as data, np.load(LOCAL, allow_pickle=False) as other:
        for key, value in zip(("subject", "session", "interface", "ch_names", "onset"),
                              (subject, session, interface, names, onset), strict=True):
            np.testing.assert_array_equal(other[key], value)
        nominal, local = data["X"], other["X"]
    assert nominal.shape == local.shape == (2130, 41, 480) and nominal.dtype == local.dtype == np.float32
    assert len(set(names.tolist())) == 41 and set(SITES).issubset(names)
    kept = np.isin(names, SITES)
    np.testing.assert_array_equal(nominal[:, kept], local[:, kept])
    assert np.isfinite(nominal).all() and np.isfinite(local).all()
    assert np.abs(nominal[:, ~kept] - local[:, ~kept]).max() > 1
    return nominal, local


def parent_seal():
    path = PARENT / "before_labels.json"
    pin(path, PARENT_SEAL)
    proof = json.loads(path.read_text())
    assert not proof["held_labels_opened"] and proof["archive"] == TRIAL and proof["stock"] == STOCK
    assert proof["actors"] == ACTOR_LOCK and len(proof["pins"]) == 9
    pin(PARENT / "source.sha256", proof["source"])
    return proof


def parent(index, who, plans, seal):
    path = PARENT / f"person_{index}.json"
    assert not path.stat().st_mode & 0o222
    pin(path, seal["pins"][path.name])
    card = json.loads(path.read_text())
    assert card["passed"] and not card["held_labels_opened"] and card["who"] == who and card["index"] == index
    assert card["archive"] == TRIAL and card["stock"] == STOCK and card["actors"] == ACTOR_LOCK
    assert card["source"] == seal["source"] and "RTX 2080 Ti" in card["worker"]
    streams = {}
    for seed in SEEDS:
        for visit in ("1", "2"):
            key = f"{who}|{visit}|{seed}"
            rows = plans[seed][f"{who}|{visit}"]
            cell = card["results"][key]
            assert np.array_equal(rows, cell["rows"])
            labels = np.asarray(cell["labels"], dtype=np.int8)
            assert labels.shape == (3, len(rows)) and np.isin(labels, (0, 1, 2)).all()
            np.testing.assert_array_equal(labels[0], labels[1])
            p = cell["proof"]
            assert p["calls"] == len(rows) and p["warm_rows"] and p["csp_fusions"]
            for field in ("sha_pupils_exact", "exact_private_tail", "exact_same_input_prefix",
                          "exact_stock_and_candidate_native_inputs"):
                assert p[field] is True
            assert all(x["windows"] == len(rows) and x["recognized"] == -1 and x["errors"] == x["ft_over"] == 0
                       for x in cell["work"])
            streams[key] = rows, labels, cell["waveform"]
    assert set(streams) == set(card["results"])
    return streams


def select(gap, labels):
    assert gap.shape == (labels.shape[1], 3) and np.isfinite(gap).all()
    stock, _, hard = labels
    present = stock != hard
    present[0] = False
    age = np.arange(len(hard))
    lazy = np.maximum(0, age - 1)
    current = present & (gap[age, hard] < gap[age, stock])
    sham = present & (gap[lazy, hard] < gap[lazy, stock])
    decisions = np.vstack((labels, hard, np.where(sham, stock, hard), np.where(current, stock, hard)))
    assert decisions.shape == (len(ARMS), len(hard)) and np.isin(decisions, (0, 1, 2)).all()
    assert (decisions[2:, 0] == hard[0]).all()
    np.testing.assert_array_equal(decisions[2], decisions[3])
    return decisions, present
