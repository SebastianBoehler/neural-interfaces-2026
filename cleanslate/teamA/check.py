"""Exercise the exported solver on real, unlabeled PROTEUS streams."""
import copy
import sys
from pathlib import Path

import numpy as np
import torch

from package import load_source


def main(package, data):
    d = np.load(data, allow_pickle=False)
    x = torch.from_numpy(d["X"])
    source = load_source(Path(package) / "submission.py")
    template = source.Solver().load_model(dict(
        submission_dir=package, ch_names=d["ch_names"], n_chans=41, n_times=480, sfreq=120, device="cuda",
    ))
    subject = d["subject"]
    rng = np.random.default_rng([71000, 0, 0])
    a, b = [rng.permutation(np.flatnonzero((subject == p) & (d["session"] == "1")))[:40]
            for p in np.unique(subject)[:2]]

    def run(m, rows):
        return [int(m.predict(x[i:i + 1].cuda())[0]) for i in rows]

    frozen = {name: value.detach().clone() for name, value in template.net.state_dict().items()}
    left, right = copy.deepcopy(template), copy.deepcopy(template)
    expected_a, expected_b = run(left, a), run(right, b)
    assert left.age == right.age == 40 and template.age == 0
    for name, value in template.net.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0, msg=f"Shared weight changed: {name}")
    assert left.adapter.delta.abs().max().item() > 0
    assert not np.array_equal(left.covariance_sum, right.covariance_sum)
    left.reset_state()
    right.reset_state()
    actual_a, actual_b = [], []
    for i, j in zip(a, b):
        actual_a += run(left, [i])
        actual_b += run(right, [j])
    assert expected_a == actual_a and expected_b == actual_b
    assert set(actual_a + actual_b) <= {0, 1, 2}
    assert run(copy.deepcopy(left), a[:12]) == expected_a[:12]
    print("SMOKE OK: updates, reset, prefix, interleaved sessions and single-label output", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
