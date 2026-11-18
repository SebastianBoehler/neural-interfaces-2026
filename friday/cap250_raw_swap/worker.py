"""Exercise the exact cap250 worker with one native raw member substituted."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from data import permutation
from slot import MEMBER, install


PACKAGE = Path("/package")
ARTIFACTS = Path("/actors")
ORDERS = (411, 1411)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(template, eeg, rows, arm, actor=None, *, keep_posteriors=False):
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    slot = install(model, actor) if arm != "stock" else None
    labels, posteriors = [], []
    for row in rows:
        window = eeg[int(row):int(row) + 1].to("cuda:0")
        label = model.predict(window)
        if label.device.type != "cpu" or label.dtype != torch.int64 or label.shape != (1,):
            raise ValueError("expected one CPU int64 label")
        if model.dead or model.cp.dead or model.cf.dead:
            raise ValueError("cap250 entered a fallback")
        posterior = model.trace["v6"]["members"][MEMBER]
        if (not np.isfinite(posterior).all() or np.any(posterior < 0)
                or abs(float(posterior.sum()) - 1) > 1e-6):
            raise ValueError("the full worker received an invalid member posterior")
        if keep_posteriors:
            posteriors.append(posterior.tolist())
        labels.append(int(label[0]))
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    activity = replay.activity(model)
    calls = slot.calls if slot is not None else 0
    if (activity["errors"] or activity["windows"] != len(rows) or activity["ft_over"]
            or (slot is not None and (calls != len(rows) or slot.window is not None))
            or not np.isin(labels, (0, 1, 2)).all()):
        raise ValueError(f"unexpected exact-worker activity for {arm}: {activity}")
    return dict(labels=labels, activity=activity, actor_calls=calls, seconds=seconds), posteriors


def stock_parity(got, expected, label):
    for key in ("labels", "activity"):
        if got[key] != expected[key]:
            raise ValueError(f"archive {key} parity failed for {label}")


def assert_prefix(long, short):
    (full, full_q), (prefix, prefix_q) = long, short
    if prefix["labels"] != full["labels"][:len(prefix_q)] or prefix_q != full_q[:len(prefix_q)]:
        raise ValueError("the current-window actor lost exact prefix or reset parity")


def sessions_for(subject, visit, who, archive):
    groups = np.char.add(np.char.add(subject, "|"), visit)
    sessions = []
    for order in ORDERS:
        schedule = permutation(groups, None, "full", order)
        for day in sorted(set(visit[subject == who])):
            key = f"{who}|{day}|{order}"
            rows = schedule[(subject == who) & (visit == day)]
            if rows.tolist() != archive[key]["rows"]:
                raise ValueError(f"matched §411b delivery changed for {key}")
            sessions.append((key, rows))
    return sessions


def control(template, eeg, sessions, archive):
    name, rows = sessions[0]
    rows = rows[:120]
    expected = archive[name]
    if name != "Scherer2015/G|1|411" or len(expected["rows"]) != 120:
        raise ValueError(f"expected the registered Scherer smoke, got {name}/{len(rows)}")
    stock = run(template, eeg, rows, "stock", keep_posteriors=True)
    identity = run(template, eeg, rows, "identity", keep_posteriors=True)
    stock_parity(stock[0], expected, name)
    stock_parity(identity[0], stock[0], "identity")
    if identity[1] != stock[1]:
        raise ValueError("the identity hook changed the native posterior")
    canonical = np.sort(sessions[0][1])[:31]
    canonical_stock = run(template, eeg, canonical, "stock", keep_posteriors=True)
    canonical_identity = run(template, eeg, canonical, "identity", keep_posteriors=True)
    stock_parity(canonical_identity[0], canonical_stock[0], "recording-order identity")
    if canonical_identity[1] != canonical_stock[1]:
        raise ValueError("the recording-order identity posterior changed")
    return stock, identity, rows


def smoke(template, eeg, sessions, archive, actors):
    stock, identity, rows = control(template, eeg, sessions, archive)
    report = {"stock": stock[0], "identity": identity[0]}
    other = sessions[1][1][:31]
    for arm, actor in actors.items():
        repeat = run(template, eeg, rows[:31], arm, actor, keep_posteriors=True)
        run(template, eeg, other, arm, actor)
        long = run(template, eeg, rows, arm, actor, keep_posteriors=True)
        assert_prefix(long, repeat)
        prefix = run(template, eeg, rows[:31], arm, actor, keep_posteriors=True)
        assert_prefix(long, prefix)
        movement = float(np.max(np.abs(np.asarray(long[1]) - np.asarray(stock[1]))))
        if movement <= 0.01:
            raise ValueError(f"{arm} emitted practically unchanged posteriors")
        report[arm] = dict(**long[0], max_member_movement=movement,
                           changed=int(np.count_nonzero(np.asarray(long[0]["labels"]) != stock[0]["labels"])),
                           exact_prefix=True)
    return report


def produce(index, template, eeg, sessions, archive, actors):
    variants = dict(stock=None, **actors)
    arms = list(variants)
    for arm in arms:
        run(template, eeg, sessions[0][1][:31], arm, variants[arm])
    output = {}
    for j, (key, rows) in enumerate(sessions):
        rotation = (index + j) % len(arms)
        result = {}
        for arm in arms[rotation:] + arms[:rotation]:
            result[arm], _ = run(template, eeg, rows, arm, variants[arm])
        stock_parity(result["stock"], archive[key], key)
        output[key] = dict(rows=rows.tolist(), arms=result)
    return output


def main(mode, index):
    if torch.__version__ != "2.14.1+cu130" or "A100" not in torch.cuda.get_device_name():
        raise ValueError("expected pinned worker Torch on A100")
    with np.load("/external.npz", allow_pickle=False) as data:
        eeg = torch.from_numpy(data["X"])
        people, visits, channels = (data[name] for name in ("subject", "session", "ch_names"))
    if eeg.shape != (2130, 41, 480) or eeg.dtype != torch.float32:
        raise ValueError("unexpected Scherer tensor")
    who = sorted(set(people))[index]
    archive_path = Path(f"/prior/cap_{index}.json")
    archive = json.loads(archive_path.read_text())
    template = replay.load(PACKAGE, channels.tolist())
    if template.perm is not None or not torch.isfinite(eeg).all():
        raise ValueError("worker channel order or finite values differ from the F16e input")
    sessions = sessions_for(people, visits, who, archive)
    paths = dict(line.split()[::-1] for line in Path("actors.sha256").read_text().splitlines()) \
        if mode != "zero" else {}
    actors, artifacts = {}, {}
    for name, expected in paths.items():
        path = ARTIFACTS / name
        arm = path.stem
        if path.name not in ("seed0.pt", "seed1.pt") or sha(path) != expected:
            raise ValueError("unexpected or unauthenticated actor")
        actors[arm] = torch.jit.load(str(path), map_location="cuda:0").eval()
        artifacts[arm] = expected
    if mode != "zero" and "seed0" not in actors:
        raise ValueError("primary seed 0 is required")
    if mode == "produce":
        payload = dict(sessions=produce(index, template, eeg, sessions, archive, actors))
        name = f"person_{index}"
    else:
        payload = dict(smoke=smoke(template, eeg, sessions, archive, actors))
        name = mode
    if "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text():
        raise ValueError("worker mapped a host environment dependency")
    receipt = dict(index=index, source=sha("source.sha256"),
                   archive=sha(archive_path), artifacts=artifacts, **payload)
    result = Path(name + ".json")
    with result.open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, allow_nan=False)
    with result.with_suffix(".sha256").open("x") as handle:
        handle.write(sha(result) + "\n")
    print(f"PASS F-C13 {name} {who}: {sha(result)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("zero", "smoke", "produce"))
    parser.add_argument("person", type=int, choices=range(9))
    args = parser.parse_args()
    main(args.mode, args.person)
