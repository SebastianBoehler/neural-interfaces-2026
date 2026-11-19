"""Produce blind Scherer banks using NI-V deliveries and native cap250 stages."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from make import ARMS, digest
from native import bitwise, fresh_tail, load, replay_tail, stream
from orders import SCHEMES, SEEDS, permutation, runs_from_manifest, stamp


DATA_SHA = "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"
RUNS_SHA = "8ea44213aa0add5058312777553f4fa527720381f473f6542cba42eee2dac10d"
ORDERS_SHA = "2de1b1a0893d31b659fa29c5b74395fa10d10c1c9005e490aeb6730af2627162"


def windows(data, manifest):
    if digest(data) != DATA_SHA or digest(manifest) != RUNS_SHA:
        raise ValueError("Unexpected Scherer data or run manifest")
    with np.load(data, allow_pickle=False) as z:
        x = torch.from_numpy(z["X"])
        subject, session, channels = z["subject"], z["session"], z["ch_names"].tolist()
    run = runs_from_manifest(manifest, subject, session)
    visit = np.char.add(np.char.add(subject, "|"), session)
    order = np.stack([permutation(visit, run, s, k) for s in SCHEMES for k in SEEDS]).astype("<i8")
    if stamp(order) != ORDERS_SHA:
        raise ValueError("Scherer delivery bank differs from the pinned NI-V permutations")
    np.testing.assert_array_equal(visit[order], np.broadcast_to(visit, order.shape))
    np.testing.assert_array_equal(run[order[len(SEEDS):]], np.broadcast_to(run, order[len(SEEDS):].shape))
    return x, subject, visit, channels, order.reshape(len(SCHEMES), len(SEEDS), -1)


def bank_case(package, templates, tails, batch):
    if len(batch) > 250:
        raise ValueError("Short-session exact reuse is invalid after age 249")
    stock = stream(templates["stock"], batch)
    known, _ = replay_tail(tails["off"], stock["x"], stock["raw"], stock["live"], False)
    bitwise(stock["terminal"], known, "off on live stock stages")
    output = dict(stock=(stock["labels"], stock["meta"]))
    for arm in ARMS[1:]:
        reversible = arm in ("B2", "B5")
        if arm == "B2" and stock["meta"]["recognition_returns"]:
            if arm not in templates:
                templates[arm] = load(package / "B2.zip", templates["channels"], str(batch.device))
            changed = stream(templates[arm], batch)
        elif arm == "B2":
            changed = stock
        if arm == "B4":
            output[arm] = stock["labels"], dict(stock["meta"], cap=320)
            continue
        live = changed if reversible else stock
        scores, work = replay_tail(tails[arm], live["x"], live["raw"], live["live"], reversible)
        if arm == "B2":
            bitwise(scores, live["terminal"], "B2 exact downstream path")
        output[arm] = scores.argmax(1).astype(np.int8), dict(live["meta"], **(work | {"cap": 320 if arm == "B5" else 250}))
    return output


def main(package, data, manifest, output, index):
    x, people, visits, channels, schedule = windows(data, manifest)
    person, delivery = divmod(int(index), len(SCHEMES) * len(SEEDS))
    scheme, seed = divmod(delivery, len(SEEDS))
    who = sorted(set(people))[person]
    device = "cuda:0"
    templates = dict(stock=load(package / "stock.zip", channels, device), channels=channels)
    tails = {arm: fresh_tail(package / f"{arm}.zip") for arm in ("off", "B1", "B2", "B3", "B5")}
    plan = schedule[scheme, seed]
    result, receipt = {}, {}
    for visit in sorted(set(visits[people == who])):
        rows = plan[visits[plan] == visit]
        predictions = bank_case(package, templates, tails, x[rows].to(device))
        result[f"{visit}|rows"] = rows
        receipt[visit] = {}
        for arm, (labels, report) in predictions.items():
            result[f"{visit}|{arm}"] = labels
            receipt[visit][arm] = report
        print(f"PASS {who} {visit} {SCHEMES[scheme]}:{SEEDS[seed]} {json.dumps(receipt[visit])}", flush=True)
    result["receipt"] = np.array(json.dumps(dict(device=torch.cuda.get_device_name(), trials=receipt)))
    with output.open("xb") as fp:
        np.savez_compressed(fp, **result)


if __name__ == "__main__":
    package, data, manifest, output, index = sys.argv[1:]
    main(*map(Path, (package, data, manifest, output)), index)
