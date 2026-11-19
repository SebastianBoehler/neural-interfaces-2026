"""One-window replay of the copied cap250 terminal-output arm on exact Scherer inputs."""

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from balance import PrefixBalance
from make import STOCK_SHA, digest

sys.path.insert(0, "/official")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "2026-competition"))
from benchmark_utils.streaming import fresh_model  # noqa: E402
from orders import SEEDS, permutation, stamp  # noqa: E402

DATA_SHA = "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"


def load(archive, names, device, expected):
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        with ZipFile(archive) as z:
            z.extractall(directory)
        spec = importlib.util.spec_from_file_location(f"lane_d_{expected}", directory / "submission.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if getattr(module, "LANE_D_ARM", "stock") != expected:
            raise ValueError(f"wrong Lane D experiment in {archive}")
        return module.Solver().load_model(dict(submission_dir=directory, ch_names=names, device=device))


def stream(model, batch, batched=False):
    scores, votes = [], []
    original = model.cp.step

    def capture(x, p, output, recognised):
        q = original(x, p, output, recognised)
        scores.append(q.copy())
        trace = model.trace.get("v6", {})
        members = trace["members"]
        groups = []
        for name, ids in sorted(model.ft.slots.items()):
            weights = model.w[1 + ids]
            groups.append(np.average([members[i] for i in ids], axis=0, weights=weights))
        families = np.stack([model.trace["core1"], *groups, trace["extra"], trace["rest"]])
        votes.append(families.argmax(1))
        return q

    model.cp.step = capture
    try:
        if batched:
            labels = model.predict(batch)
        else:
            labels = torch.cat([model.predict(row[None]) for row in batch])
    finally:
        del model.cp.step
    if labels.dtype != torch.int64 or labels.device.type != "cpu" or labels.shape != (len(batch),):
        raise ValueError("cap250 must emit one int64 CPU label per window")
    for name, stage in (("chain", model), ("route", model.route), ("routeB", model.routeB),
                        ("count", model.cf), ("csp", model.cp)):
        log = stage.log6 if name == "chain" else stage.log
        if log["err"]:
            raise RuntimeError(f"{name} error: {log['first']}")
    counts = [model.log6["n"], model.log6["rec"], model.ft.refits, model.ft.steps,
              model.cf.t, model.cp.fused, model.cp.refits]
    return np.stack(scores), labels.numpy(), np.stack(votes), np.array(counts)


def expected(scores):
    state = PrefixBalance()
    return np.array([state.step(row).argmax() for row in scores], dtype=np.int64)


def identical(a, b):
    for left, right in zip(a, b):
        if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
            raise ValueError(f"cap250 parity differs for shape {left.shape}")


def smoke(stock, off, candidate, names, batch):
    if digest(stock) != STOCK_SHA:
        raise ValueError("stock archive changed")
    device = str(batch.device)
    original = load(stock, names, device, "stock")
    no_change = load(off, names, device, "off")
    before = stream(fresh_model(original), batch)
    identical(before, stream(fresh_model(no_change), batch))
    model = fresh_model(candidate)
    treatment = stream(model, batch)
    identical((before[0], *before[2:]), (treatment[0], *treatment[2:]))
    np.testing.assert_array_equal(treatment[1], expected(before[0]))
    if np.array_equal(treatment[1], before[1]):
        raise ValueError("smoke did not exercise a changed decision")
    model.reset_state()
    prefix = stream(model, batch[:25], batched=True)
    identical(tuple(part[:25] for part in treatment[:3]), prefix[:3])
    print("PASS exact stock/off bytes, D-c upstream and work parity, reset/prefix, batch and actual output changes")


def main(directory, data, item, mode, device):
    if digest(data) != DATA_SHA:
        raise ValueError("Scherer input differs")
    with np.load(data, allow_pickle=False) as z:
        x = torch.from_numpy(z["X"])
        subject, session, names = z["subject"], z["session"], z["ch_names"].tolist()
    visit = np.char.add(np.char.add(subject, "/"), session)
    plans = np.stack([permutation(visit, None, "full", seed) for seed in SEEDS]).astype("<i8")
    if stamp(plans) != "742fbfd6218b48404658b8342ee4d93ee45ea0b2bdf12bb24856f5b9fba53a58":
        raise ValueError("full delivery differs from locked NI-V draws")
    person, order = divmod(int(item), len(SEEDS))
    who = sorted(set(subject))[person]
    selected = plans[order][subject[plans[order]] == who]
    candidate = load(directory / "zips/output.zip", names, device, "output")
    if mode in {"smoke", "local-smoke"}:
        rows = selected[visit[selected] == visit[selected[0]]]
        if mode == "local-smoke":
            rows = rows[:44]
        smoke(directory / "stock.zip", directory / "zips/off.zip", candidate, names, x[rows].to(device))
        return
    if mode != "run":
        raise ValueError(f"invalid replay mode: {mode}")
    result = {}
    for key in sorted(set(visit[selected])):
        rows = selected[visit[selected] == key]
        scores, labels, votes, counts = stream(fresh_model(candidate), x[rows].to(device))
        stock = scores.argmax(1)
        np.testing.assert_array_equal(labels, expected(scores))
        if counts[0] != len(rows) or counts[4] != len(rows):
            raise ValueError(f"incomplete session stream: {key}")
        for name, value in dict(rows=rows, scores=scores, labels=labels, stock=stock, votes=votes, counts=counts).items():
            result[f"{key}|{name}"] = value
        print(f"PASS {key} seed {SEEDS[order]} rows {len(rows)} changed {(labels != stock).sum()} work {counts}", flush=True)
    with (directory / "banks" / f"{item}.npz").open("xb") as fp:
        np.savez_compressed(fp, **result)


if __name__ == "__main__":
    directory, data, item, mode, device = sys.argv[1:]
    main(Path(directory), Path(data), item, mode, device)
