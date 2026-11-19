"""Fit only the two small readouts from authenticated first visits."""

import json
from pathlib import Path

import numpy as np
import torch

import benchmark_utils.base_solver
from assets import digest, sha
from geometry import EEG, build, forward_feature, live_waveform, row_labels, waveform_proof
from inputs import bundle_and_fit
from record import load_model


def fit(who, eeg, people, visits, channels):
    folder, source, _, _ = bundle_and_fit(who, people, visits)
    provenance = waveform_proof(who, folder, people, visits)
    model = load_model(folder, channels, f"prototype_fit_{who}", "cpu")
    stream, actor = live_waveform(model)
    rows = np.flatnonzero(visits == "1")
    assert not np.isin(rows, source["held"]).any()
    labels = row_labels(EEG, rows)
    feature = np.empty((len(rows), 0))
    with torch.no_grad():
        for j, person in enumerate(sorted(set(people))):
            i = np.flatnonzero(people[rows] == person)
            assert len(i) >= 175
            stream.reset_state()
            actor.reset_state()
            values = []
            for k in i:
                window = stream(eeg[rows[k]])
                z, _ = forward_feature(actor, torch.from_numpy(window[None]).to(actor.dtype), {})
                values.append(z.numpy()[0])
            h = np.asarray(values, dtype=np.float64)
            if j == 0:
                feature = np.empty((len(rows), h.shape[1]), dtype=np.float64)
            feature[i] = h
        focus = np.flatnonzero(people[rows] == f"Dreyer2026Proteus/{who}")[:31]
        stream.reset_state()
        actor.reset_state()
        reference = []
        for k in focus:
            x = torch.from_numpy(stream(eeg[rows[k]])[None]).to(actor.dtype)
            reference.append(actor.proba(x, {}).numpy()[0])
        stream.reset_state()
        actor.reset_state()
        for k, p in zip(focus, reference, strict=True):
            x = torch.from_numpy(stream(eeg[rows[k]])[None]).to(actor.dtype)
            z, q = forward_feature(actor, x, {})
            np.testing.assert_array_equal(z.numpy()[0], feature[k])
            np.testing.assert_array_equal(q.numpy()[0], p)
    result = build(feature, labels, people[rows], rows, who)
    assert all(not np.isin(result[k], source["held"]).any() and np.isin(result[k], rows).all()
               for k in ("own_rows", "population_rows", "scale_rows"))
    assert len(result["own_rows"]) == len(result["population_rows"])
    path = Path(f"{who}.npz")
    with path.open("xb") as f:
        np.savez_compressed(f, **result)
    path.chmod(0o444)
    receipt = dict(who=who, file_sha=sha(path), feature_dim=feature.shape[1], fold_view=source["fit_sha"],
                   actor_proof=provenance, first_visit_global_sha=digest(rows), source_labels_only=True,
                   exact_feature_prefix=True, exact_native_probabilities=True, counts=result["class_counts"].tolist(),
                   scopes={k: digest(result[k]) for k in ("own_rows", "population_rows", "scale_rows")})
    path = path.with_suffix(".json")
    with path.open("x") as f:
        json.dump(receipt, f, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("F-C16 source geometry passed", who, sha(path), receipt, flush=True)


def main():
    assert torch.__version__ == "2.14.1+cu130" and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with np.load(EEG, allow_pickle=False) as f:
        eeg, people, visits, ch = (f[k] for k in ("X", "subject", "session", "ch_names"))
    assert eeg.shape == (2655, 41, 480) and eeg.dtype == np.float32
    for who in ("P10", "P04"):
        fit(who, eeg, people, visits, ch.tolist())


if __name__ == "__main__":
    main()
