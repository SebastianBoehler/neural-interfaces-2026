"""Match the scored stock decisions, role coverage and recognition logs to the independent production receipts."""
import json
import re
from pathlib import Path

import numpy as np


def coverage(group, records):
    expected = {"return_pub": 16, "new_pub": 56, "return_sch": 72, "single_sch": 72, "new_sch": 144,
                "publiconly_sch": 144, "wrong_sch": 144, "return_net": 456, "new_net": 456, "single_net": 456}
    if len(records) != expected[group]:
        raise ValueError(f"{group} has {len(records)} stream-orders, expected {expected[group]}")
    if group != "wrong_sch":
        assert len({(r["person"], r["day"], r["order"]) for r in records}) == len(records)


def public_reference(stock, records, role):
    receipt = json.loads((stock / "stock/scores.json").read_text())
    jobs = [json.loads(line) for line in (stock / "jobs.jsonl").read_text().splitlines()]
    routed = {}
    for person in sorted({r["person"][4:] for r in records}):
        job, = {j["job"] for j in jobs if j["node"] == f"stock.{role}_{person}.stream"}
        file, = (stock / "logs").glob(f"*-{job}_*.out")
        body = file.read_text()
        passes = re.findall(r"^PASS (P\d+) (\d+) (\d+) (\d+)", body, re.M)
        status = re.findall(r"^\[v6\] windows (\d+) .*?recognised_at (-?\d+) .*?errors (\d+)", body, re.M)
        for (p, seed, day, n), (seen, at, err) in zip(passes, status, strict=True):
            assert p == person and n == seen and int(err) == 0
            routed[p, int(seed), int(day)] = int(at)
    assert len(routed) == len(records) and (role != "new" or max(routed.values()) == -1)
    for seed in range(71000, 71004):
        rows = [r for r in records if r["order"] == ("full", seed)]
        actual = np.concatenate([r["ba"]["stock"] for r in rows])
        official = receipt[f"{role}|{seed}"]
        assert len(actual) == official["n_cells"]
        np.testing.assert_allclose(actual.mean(), official["cell_mean"], atol=1e-10, rtol=0)
        for r in rows:
            assert (r["person"][4:], seed, r["day"]) in routed
    return sorted(n+1 for n in routed.values() if n >= 0)
