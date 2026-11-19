"""Pin the fresh Scherer delivery and old complete-worker controls."""

import hashlib
import json
from pathlib import Path

import numpy as np

from data import permutation


SEEDS = (2411, 3411)
ORDERS = {
    411: "fc5fb76599716457329327e80e9982bcc02360860c824f692f8880ec72e9814a",
    1411: "a4e996f5aa150227875aae7e624d7bb29c6618ecfcda9cffedbfff480ddacdbe",
    2411: "d1ff01b3d55cc59b4877843ed81de0e287d4a4c42153c4d439289b7668b05905",
    3411: "f3cf6417dc6a773d14dd4ea28a924a7bb1b9b1045a0019fba43b362ef2c551cf",
}
ACTOR = "0dc677091668ad8acc6716a9bd75df6371829aa13ac6cd90985a3ef675126cdd"
ARMS = ("stock", "directional", "cognitive", "combined")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def seal(path, result):
    path = Path(path)
    with path.open("x") as f:
        json.dump(result, f, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    return sha(path)


def schedule(people, visits):
    streams = np.char.add(np.char.add(people, "|"), visits)
    assert streams.shape == (2130,) and len(set(people)) == 9 and len(set(streams)) == 18
    output = {}
    for seed, digest in ORDERS.items():
        p = permutation(streams, None, "full", seed).astype("<i8")
        assert hashlib.sha256(p.tobytes()).hexdigest() == digest, seed
        output[seed] = {name: p[streams == name] for name in sorted(set(streams))}
    assert output[411]["Scherer2015/G|1"][0] == 1257 and output[411]["Scherer2015/G|2"][0] == 1383
    assert output[2411]["Scherer2015/G|1"][0] == 1195 and output[3411]["Scherer2015/G|1"][0] == 1297
    return output


def archive_parity(run, reference):
    assert run["labels"] == reference["labels"] and run["activity"] == reference["activity"]
