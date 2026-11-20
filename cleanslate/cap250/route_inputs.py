"""Authenticate the nested B2 posterior bank used to learn the route debias direction."""

import json

from components import Variant


def b2_fits():
    exclude = {"fbc_ea_fixed", "fbc_raw_fixed", "sinc_ea", "fbl_ea", "eye", "rest", "identity", "count"}
    return [(family, seed) for family, seed in Variant().fits() if family not in exclude]


def subfold(people, first, held):
    return tuple(p for p in people if p != held), tuple(p for p in first if p != held)


def load_oof(manifest, index, people, first, data):
    import numpy as np

    from source import file_digest, row_digest, source_rows

    by_person = json.loads(manifest.read_text())
    if set(by_person) != set(people):
        raise ValueError(f"route OOF manifest lacks required source people: {manifest}")
    data_hash = file_digest(data)
    collected = np.full((len(index), 3), np.nan)
    for held in people:
        fit, fit_first = subfold(people, first, held)
        actual_target = source_rows((held,), (held,) if held in first else (), data)
        with np.load(by_person[held], allow_pickle=False) as rows:
            metadata = json.loads(str(rows["metadata"]))
            expected = dict(source_rows=row_digest(source_rows(fit, fit_first, data)), data_sha256=data_hash,
                            query=held, query_rows=row_digest(actual_target))
            if (any(metadata.get(k) != v for k, v in expected.items())
                    or not np.array_equal(rows["index"], actual_target)):
                raise ValueError(f"route OOF bank has wrong nested source or query: {held}")
            offsets = np.searchsorted(index, actual_target)
            if not np.array_equal(index[offsets], actual_target) or np.isfinite(collected[offsets]).any():
                raise ValueError(f"route OOF row coverage mismatch: {held}")
            collected[offsets] = rows["proba"]
    if (not np.isfinite(collected).all() or (collected < 0).any()
            or not np.allclose(collected.sum(1), 1, atol=1e-8, rtol=0)):
        raise ValueError("route OOF probabilities are missing or invalid")
    return collected
