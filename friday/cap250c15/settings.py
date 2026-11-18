"""Scope and controls for the robust outer rebalancer experiment."""

import hashlib

import numpy as np

from data import permutation


ARMS = ("stock", "scalar", "robust")
SEEDS = (4411, 5411)
ORDERS = {
    411: "fc5fb76599716457329327e80e9982bcc02360860c824f692f8880ec72e9814a",
    1411: "a4e996f5aa150227875aae7e624d7bb29c6618ecfcda9cffedbfff480ddacdbe",
    4411: "b90c6b732646b5543d94fb974e46a2171b0355c3fc3c8c9426ee1e1b754072d2",
    5411: "12b85e614c816bf82ecc4aa19020f81e909a805c18d337985707059eb8fbdf61",
}
STOCK_WORK = ("windows", "recognized", "ft_refits", "ft_steps", "ft_relabels", "ft_capped", "ft_over", "errors")


def schedule(people, visits):
    streams = np.char.add(np.char.add(people, "|"), visits)
    assert streams.shape == (2130,) and len(set(people)) == 9 and len(set(streams)) == 18
    result = {}
    for seed, digest in ORDERS.items():
        rows = permutation(streams, None, "full", seed).astype("<i8")
        assert hashlib.sha256(rows.tobytes()).hexdigest() == digest
        result[seed] = {name: rows[streams == name] for name in sorted(set(streams))}
    return result
