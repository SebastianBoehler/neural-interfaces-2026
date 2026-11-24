"""Source-locked training rows and causal input alignment for the D1 REVE experiment."""
import numpy as np

from oracle import DATA_SHA, ORDER_SHA, require, sha

PEOPLE = ("P02", "P05", "P08", "P10")
ALL = tuple(f"P{i:02}" for i in range(1, 11))
FULL_ARMS = "BC"
FULL_SEEDS = 3
FULL_ROLES = (*ALL, "returning", "all")
TRAIN_ORDERS = ("full0", "full1")
FULL_ORDERS = tuple(f"full{i}" for i in range(8))
EXTERNAL = {
    "shin2017a": (("left_hand", "right_hand"), (0, 0), (2, 4)),
    "shin2017b": (("rest", "subtraction"), (3, 1), None),
    "shin2018": (("WG", "baseline"), (2, 3), None),
    "physionetmi": (("feet", "hands", "left_hand", "rest", "right_hand"), (0, 0, 0, 3, 0), (0, 1, 2, -1, 4)),
    "dreyer2023": (("left_hand", "right_hand"), (0, 0), (2, 4)),
    "scherer2015": (None, (0, 1, 2), None),
}


def fold_rows(person, visits, held):
    return_seen = np.isin(person, ("P04", "P10")) & (visits != "1")
    query = return_seen if held == "returning" else person == held
    if held == "all":
        query[:] = False
    require(set(person) == set(ALL), "unexpected PROTEUS roster")
    require(held == "all" or query.any(), "empty query")
    return np.flatnonzero(~query), np.flatnonzero(query)


def align(x, stream, order, prior=None):
    """Normalize each window from the prefix ending at that window, including itself."""
    x = x.astype(np.float64)
    x -= x.mean(-1, keepdims=True)
    cov = x @ x.transpose(0, 2, 1) / x.shape[2]
    k = 5 if prior is not None else 0
    out = np.empty(x.shape, np.float32)
    for key in dict.fromkeys(stream[order]):
        ix = order[stream[order] == key]
        mean = (np.cumsum(cov[ix], axis=0) + (k * prior if k else 0))
        mean /= (k + np.arange(1, len(ix) + 1))[:, None, None]
        s, u = np.linalg.eigh(mean)
        s = np.maximum(s, np.trace(mean, axis1=1, axis2=2)[:, None] * 1e-5 / x.shape[1])
        out[ix] = (((u * (s ** -0.5)[:, None, :]) @ u.transpose(0, 2, 1)) @ x[ix]).astype(np.float32)
    require(len(order) == len(x) and np.isfinite(out).all(), "invalid aligned source")
    return out


def public(root, held, full=False):
    path, orders = root / "data/proteus_windows.npz", root / "runs/shuffle_perms.npz"
    require(sha(path) == DATA_SHA and sha(orders) == ORDER_SHA, "PROTEUS data or order hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        person = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]])
        visit = z["session"]
        train, query = fold_rows(person, visit, held)
        stream = np.char.add(np.char.add(person, "|"), visit)
        x, names = z["X"], z["ch_names"].tolist()
        labels = z["y"][train]
    centered = x[train].astype(np.float64)
    centered -= centered.mean(-1, keepdims=True)
    prior = np.einsum("nct,ndt->cd", centered, centered, optimize=True) / (len(train) * x.shape[-1])
    rng = np.random.default_rng(20261119)
    trained = [align(x[train], stream[train], rng.permutation(len(train)), prior) for _ in range(2)]
    role = "seen_session" if held == "returning" else "all_loso"
    queries = {}
    with np.load(orders, allow_pickle=False) as p:
        for name in FULL_ORDERS if full else TRAIN_ORDERS:
            rows = p[f"{name[:-1]}|{name[-1]}|{role}"]
            rows = rows[np.isin(rows, query)]
            local = np.searchsorted(query, rows)
            np.testing.assert_array_equal(query[local], rows)
            if len(rows):
                np.testing.assert_array_equal(np.sort(rows), query)
                queries[name] = (rows, align(x[query], stream[query], local, prior))
    if queries:
        rows, normalized = queries["full0"]
        first = rows[stream[rows] == stream[rows[0]]][:8]
        expected = normalized[np.searchsorted(query, first)]
        observed = align(x[first], stream[first], np.arange(len(first)), prior)
        np.testing.assert_allclose(expected, observed, rtol=0, atol=2e-6)
    return trained, labels, train, query, queries, names, prior


# The Galvani §351 manifest verified all original raw-to-export labels: HAND 0, SUB 1, WORD 2.
SCHERER_EXPORT = "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"


def external_source(root, corpus, canary=False):
    expected, task_map, subtype_map = EXTERNAL[corpus]
    folder = root / "data" / corpus
    files = sorted(folder.glob("windows_*.npz"))
    if not files:
        files = [folder / "windows.npz"]
    require(all(f.is_file() for f in files), f"missing {corpus} windows")
    if canary:
        files = files[:1]
    rng = np.random.default_rng(19112026)
    waves, labels, subtypes, domains, pins = [], [], [], [], {}
    montages = set()
    for path in files:
        digest = sha(path)
        pins[str(path.relative_to(root))] = digest
        if corpus == "scherer2015":
            require(digest == SCHERER_EXPORT, "Scherer export must match the verified HAND/SUB/WORD bank")
        with np.load(path, allow_pickle=False) as z:
            encoded_tasks, encoded_subtypes = task_map, subtype_map
            if expected is not None:
                names = tuple(z["classes"])
                require(len(names) == len(expected) and set(names) == set(expected),
                        f"{corpus} source label names changed: {names}")
                indexes = [expected.index(name) for name in names]
                encoded_tasks = np.asarray(task_map)[indexes]
                if subtype_map is not None:
                    encoded_subtypes = np.asarray(subtype_map)[indexes]
            y, person, visit = z["y"], z["subject"], z["session"]
            stream = np.char.add(np.char.add(person, "|"), visit)
            available = [c for c in z["ch_names"] if c not in z["interpolated"]]
            montages.add(tuple(available))
            col = [z["ch_names"].tolist().index(c) for c in available]
            eligible = np.ones(len(y), bool)
            if corpus == "shin2018":
                eligible = z["win"] == 0
            selected = []
            for key in np.unique(person):
                for label in np.unique(y[person == key]):
                    rr = np.flatnonzero((person == key) & (y == label) & eligible)
                    selected.extend(rng.choice(rr, min(len(rr), 20), replace=False))
            rr = np.array(sorted(selected))
            x = z["X"][rr][:, col, :240 if corpus == "physionetmi" else 480]
            # Domain-specific referencing: restrict to true electrodes and align each source visit.
            x = align(x, stream[rr], rng.permutation(len(rr)))
            waves.append(x)
            labels.append(np.asarray(encoded_tasks)[y[rr]])
            subtypes.append(np.asarray(encoded_subtypes)[y[rr]] if encoded_subtypes is not None else np.full(len(rr), -1))
            domains.extend(person[rr])
    require(len(montages) == 1, f"inconsistent native montage for {corpus}")
    return np.concatenate(waves), np.concatenate(labels), np.concatenate(subtypes), list(montages.pop()), sorted(set(domains)), pins
