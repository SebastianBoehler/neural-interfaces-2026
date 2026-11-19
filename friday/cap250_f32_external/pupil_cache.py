"""Generate opaque all-source hard, detached and joint CPU pupil banks."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from fixture import EXPORTS, actor_lock, chosen
from settings import sha, stamp


@torch.no_grad()
def session(templates, eeg, rows):
    poses = [copy.deepcopy(template) for template in templates]
    data = []
    with torch.jit.optimized_execution(False):
        assert all(a.reset().item() == 0 for a in poses)
        for row in rows:
            x = eeg[int(row):int(row)+1].contiguous()
            y = [p(x) for p in poses]
            assert all(q.shape == (1, 3) and q.dtype == torch.float64 for q in y)
            data.append(np.stack(tuple(q.cpu().numpy()[0] for q in y)))
    bank = np.asarray(data).reshape(len(rows), len(EXPORTS), 2, 3)
    assert np.isfinite(bank).all() and (bank >= 0).all()
    np.testing.assert_allclose(bank.sum(-1), 1., rtol=0, atol=1e-12)
    np.testing.assert_array_equal(bank[:, 0], bank[:, 1])
    return bank


def run(mode, index):
    assert torch.__version__ == "2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    lock = actor_lock("actors.json")
    assert lock["device"] == "cpu" and len(lock["export_receipts"]) == 2
    templates = []
    for kind in EXPORTS:
        for seed, pinned in enumerate(lock["actors"][kind]):
            path = Path("/actors") / kind / f"raw_s1c_s{seed}.pt"
            assert sha(path) == pinned
            templates.append(torch.jit.load(str(path), map_location="cpu").eval())
    choices, _, _, who, _ = chosen(mode, index)
    result = {}
    keys = list(choices)
    for n, key in enumerate(keys):
        eeg, rows = choices[key]
        post = session(templates, eeg, rows)
        other_x, other_rows = choices[keys[(n+1) % len(keys)]]
        session(templates, other_x, other_rows[:17])
        np.testing.assert_array_equal(post[:17], session(templates, eeg, rows[:17]))
        result[key+"|p"] = post
        result[key+"|rows"] = rows
        result[key+"|x"] = np.array(stamp(eeg[rows].numpy()))
    meta = dict(source=sha("source.sha256"), mode=mode, index=index, who=who, actors=sha("actors.json"),
                keys=keys, exact_detach=True, reset_prefix=True, held_labels_opened=False)
    path = Path(f"pupils_{mode}_{index}.npz")
    with path.open("xb") as file:
        np.savez_compressed(file, metadata=json.dumps(meta, sort_keys=True), **result)
    path.chmod(0o444)
    print("PASS F32x native pupil bank", mode, index, sha(path), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "produce"))
    parser.add_argument("index", type=int, choices=range(9))
    args = parser.parse_args()
    run(args.mode, args.index)
