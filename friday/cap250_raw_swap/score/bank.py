"""Authenticate exact cap250 and F-C13 predictions without reading gold."""

from collections import Counter
import hashlib
import json
import math
import os

import numpy as np

from data import permutation


ORDERS = (411, 1411)
ROWS = 2130


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seal_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    return sha(path)


def metadata(data, anchor):
    with np.load(data, allow_pickle=False) as bank:
        subject, day, interface = (bank[k] for k in ("subject", "session", "interface"))
    stream = np.char.add(np.char.add(subject, "|"), day)
    people = sorted(set(subject))
    if (subject.shape != (ROWS,) or len(people) != 9 or people != anchor["people"]
            or any(set(day[subject == person]) != {"1", "2"} for person in people)
            or any(len(set(interface[stream == session])) != 1 for session in set(stream))):
        raise ValueError("unexpected Scherer people, visits or official cells")
    return subject, day, dict(stream=stream, interface=interface)


def session_keys(person):
    return {f"{person}|{visit}|{order}" for order in ORDERS for visit in (1, 2)}


def check_arm(key, arm, data, reference, count):
    labels, activity, seconds, calls = (data[name] for name in ("labels", "activity", "seconds", "actor_calls"))
    if (set(activity) != set(reference["activity"]) or activity["windows"] != count or activity["errors"]
            or activity["ft_over"] or any(type(value) is not int for value in activity.values())):
        raise ValueError(f"fallback, work overflow or malformed counters for {key}/{arm}")
    if (len(labels) != count or any(type(label) is not int or label not in (0, 1, 2) for label in labels)
            or type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0
            or type(calls) is not int or calls != (0 if arm == "stock" else count)):
        raise ValueError(f"invalid predictions, timing or native actor call count for {key}/{arm}")
    if arm == "stock" and (labels != reference["labels"] or activity != reference["activity"]):
        raise ValueError(f"stock archive mismatch for {key}")
    return labels, seconds, activity


def locked_bank(release, prior, source, artifacts, subject, day, stream):
    if sha(release / "source.sha256") != source:
        raise ValueError("worker release source differs from the expected digest")
    arms = ("stock", *artifacts)
    roster = sorted(set(subject))
    deliveries = {order: permutation(stream, None, "full", order) for order in ORDERS}
    decisions = np.full((len(arms), len(ORDERS), ROWS), -1, dtype=np.int8)
    runtime = np.zeros((len(arms), len(roster), 2, len(ORDERS)))
    counter_changes = {arm: Counter() for arm in arms[1:]}
    receipts, archives = {}, {}
    for i, person in enumerate(roster):
        reference = prior / f"cap_{i}.json"
        archive_sha = sha(reference)
        original = json.loads(reference.read_text())
        path = release / f"person_{i}.json"
        digest = sha(path)
        if path.with_suffix(".sha256").read_text().strip() != digest:
            raise ValueError(f"worker receipt digest differs for {path}")
        raw = json.loads(path.read_text())
        if ((raw["index"], raw["source"], raw["archive"], raw["artifacts"]) != (i, source, archive_sha, artifacts)
                or raw["sessions"].keys() != original.keys() or original.keys() != session_keys(person)):
            raise ValueError(f"worker provenance or session coverage differs for {path}")
        receipts[path.name] = digest
        archives[reference.name] = archive_sha
        for key, reference_session in original.items():
            _, visit, order_name = key.split("|")
            order, v = int(order_name), int(visit) - 1
            j = ORDERS.index(order)
            rows = deliveries[order][(subject == person) & (day == visit)]
            entry = raw["sessions"][key]
            if (not len(rows) or entry["rows"] != rows.tolist() or reference_session["rows"] != rows.tolist()
                    or entry["arms"].keys() != set(arms) or (decisions[:, j, rows] != -1).any()):
                raise ValueError(f"row delivery, duplication or arm coverage differs for {key}")
            for a, arm in enumerate(arms):
                labels, seconds, activity = check_arm(key, arm, entry["arms"][arm], reference_session, len(rows))
                decisions[a, j, rows] = labels
                runtime[a, i, v, j] = seconds
                if arm != "stock":
                    counter_changes[arm].update(k for k in activity if activity[k] != reference_session["activity"][k])
    if (decisions < 0).any() or (runtime <= 0).any():
        raise ValueError("incomplete rows or timed sessions")
    return decisions, runtime, dict(
        held_labels_opened=False, source=source, artifacts=artifacts, people=roster, orders=ORDERS,
        rows_per_order=ROWS, actor_calls={arm: (0 if arm == "stock" else ROWS * len(ORDERS)) for arm in arms},
        worker_receipts=receipts, archived_stock=archives, candidate_counter_changes=counter_changes,
        decisions_sha256=hashlib.sha256(decisions.tobytes()).hexdigest(),
    )
