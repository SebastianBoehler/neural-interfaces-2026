"""Authenticate F-C20's four untouched hard-CE source states."""

import json

import numpy as np

from eye_scope import BASE, DATA_SHA, ORDER_SHA, pin, rows_sha, sha


F17 = BASE / "ni26_friday_s1c_phase_20261118"
ACTORS = BASE / "ni26_friday_return_c20_actors_v2_20261119"
ACTOR = "7ed98f96ba6b05165a1bc035a20af0d7dc516b9b1d61c82533abcc85ecbf2372"
BLIND = "77a25afbba7061dde2c781f3e8fb810b49d653aa57d09dcbc4b5abc0bb2abb37"
PINS = {
    ("P04", 0): ("531ae1224a50b99954d1e6d8fbf5b341e1ef9e5af86bd8387a33a2cd5efe11fc",
                 "5227d43d16612c0c093bfbaaeb25ee95a0edf17088006c848a8675d3fcabc342",
                 "8b35336e21c0e47d1c1d55bcd4943c5f03bedc78cbaef3a2a6d2d7e7a3a7e0f2"),
    ("P04", 1): ("13e5d06e788eada5ab9aebd31a9d4f90d3d8fb742f8ce9b546a8004eefb38e10",
                 "9ffabec8717d0c92f62f30c97c89d312dd73db2399875ace21bf5e29b2bc7022",
                 "a197881c66f4a00771b7df043e04d71442975aecc8a88ec10c91fc6f2d4bfdb2"),
    ("P10", 0): ("e47e32891f707c958524afbea0392701f001e605f8a7f9e06cedccdff2a92690",
                 "3a59c623f426e470eacf2b0aff2267a10231d7aec6d17268a6e29bf9ccedc7cb",
                 "712a8cf62605606498994e8660bb20dbae7afd1e20952445ba2838645221efb9"),
    ("P10", 1): ("463e935ed43914d86a0ab53cc21800660f552dc8436693a2e2b2bcc67f3a430b",
                 "ace372342b963191f101541b278a878847a71ff95e1f439598954fd0c193b9e1",
                 "18839c98f93d3899e98c907bd410e7265de10de7b65dfb1e03b9a90932aac1fb"),
}


def source(who, seed, subject, visit):
    pin(F17 / "actor.sha256", ACTOR)
    pin(F17 / "before_labels.sha256", BLIND)
    c, m, f = PINS[who, seed]
    checkpoint = F17 / "checkpoints" / f"hard_{seed}_{who}.pt"
    metadata = checkpoint.with_suffix(".json")
    signed = F17 / "folds" / f"{who}_s{seed}.npz"
    for path, digest in ((checkpoint, c), (metadata, m), (signed, f)):
        pin(path, digest)
    rows = np.flatnonzero(subject == f"Dreyer2026Proteus/{who}")
    train = np.flatnonzero(subject != f"Dreyer2026Proteus/{who}")
    meta = json.loads(metadata.read_text())
    checks = dict(held=who, seed=seed, epochs=40, source_rows=rows_sha(train), source_people=sorted(set(subject[train])),
                  actor=ACTOR, checkpoint=c, weights_origin=c, held_labels_opened=False, q_gradients_nonzero=0,
                  torch="2.14.1+cu126", braindecode="1.8.1", dataset=DATA_SHA, kind="hard", perturbed_rows=0)
    if (any(meta[k] != v for k, v in checks.items()) or len(checks["source_people"]) != 9
            or any(not np.isfinite(meta[k]) or meta[k] <= 0 for k in ("scale", "max_gradient", "steps"))
            or set(visit[rows]) != {"1", "2", "3"}):
        raise ValueError("authentic hard pupil failed entire-person exclusion")
    with np.load(signed, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["index"], rows)
        info = json.loads(str(z["metadata"]))
        proof = info["evidence"]["hard"]
        if (info["actor"] != ACTOR or info["held"] != who or info["seed"] != seed
                or info["held_labels_opened"] or info["order_sha"] != ORDER_SHA
                or proof["state"] != meta["state"] or proof["checkpoint"] != c or proof["receipt"] != m):
            raise ValueError("signed F17 hard decisions mismatch their source")
        labels = z["hard_bounded"][info["orders"].index("full0")]
    return checkpoint, meta, rows, labels
