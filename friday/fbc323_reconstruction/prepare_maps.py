"""Freeze the §428 reconstruction operators restricted to common source geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np


NATIVE = "AFz F7 F3 Fz F4 F8 FC3 FCz FC4 T7 C3 Cz C4 T8 CP3 CPz CP4 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO3 PO4 O1 O2".split()


def make(data, output):
    with np.load(data, allow_pickle=False) as z:
        targets = z["ch_names"].astype(str).tolist()
    keep = np.asarray([i for i, name in enumerate(targets) if name in NATIVE])
    retained = [targets[i] for i in keep]
    missing = [n for n in targets if n not in NATIVE]
    if len(targets) != 41 or len(retained) != 19 or len(missing) != 22:
        raise ValueError("the verified common measured montage differs")
    info = mne.create_info(retained + missing, 120., "eeg")
    impulse = np.concatenate((np.eye(19), np.zeros((22, 19))), axis=0)[None]
    ep = mne.EpochsArray(impulse, info, verbose=False)
    ep.set_montage("standard_1005")
    ep.info["bads"] = missing
    ep.interpolate_bads(reset_bads=True, verbose=False)
    spline = ep.get_data(picks=targets)[0].astype(np.float32)

    fields = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    coords = np.stack([fields[n] for n in retained])
    local = np.zeros_like(spline)
    for i, name in enumerate(targets):
        if name in retained:
            local[i, retained.index(name)] = 1.
        else:
            d2 = ((coords - fields[name]) ** 2).sum(1)
            select = np.argsort(d2, kind="stable")[:3]
            values = 1. / d2[select]
            local[i, select] = values / values.sum()
    for matrix in (spline, local):
        if not np.array_equal(matrix[keep], np.eye(19)) or not np.isfinite(matrix).all():
            raise ValueError("recorded sites are not preserved by the geometry")
        np.testing.assert_allclose(matrix.sum(1), 1., atol=2e-6, rtol=0)
    if np.array_equal(spline, local):
        raise ValueError("reconstruction methods did not differ")
    with output.open("xb") as handle:
        np.savez_compressed(handle, channels=np.asarray(targets), retained=np.asarray(retained), keep=keep,
                            spherical=spline, local=local)
    identity = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(dict(maps=identity, retained=retained, mne=mne.__version__, remaining=len(missing), source_only=True),
                     sort_keys=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    make(a.source, a.output)
