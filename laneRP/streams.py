"""Pair frozen production predictions with independently regenerated row orders."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "niv"))
import orders


@dataclass
class Stream:
    person: str
    day: int
    scheme: str
    seed: int
    rows: np.ndarray
    new: np.ndarray
    stock: np.ndarray
    heldnew: np.ndarray


def read_labels(archive, tag, rows, new=False):
    np.testing.assert_array_equal(archive[tag + "|rows"], rows)
    rec, routed, err, _ = archive[tag + "|meta"]
    if err or (new and (rec != -1 or routed != 0)):
        raise ValueError(f"invalid stock native route status {tag}: {rec}, {routed}, {err}")
    return archive[tag + "|labels"]


def public(args, ds, ret):
    root = args.stock
    with np.load(args.public) as p:
        subject = np.array([v.split("/")[-1] for v in p["subject"]])
        session, run = p["session"], p["run"]
    for seed in range(71000, 71004):
        for key, rows in orders.plans(subject, session, run, "full", seed).items():
            who, visit = key.split("|")
            if ret and visit == "1":
                continue
            tag = f"{seed}_{visit}"
            def labels(path):
                with np.load(path) as z:
                    np.testing.assert_array_equal(z["rows_" + tag], rows)
                    return z["labels_" + tag]
            held = labels(root / f"pred_new_{who}.npz")
            base = labels(args.population / f"pred_pop_{who}.npz") if ret else held
            stock = labels(root / f"pred_ret_{who}.npz") if ret else held
            yield Stream("pub_" + who, int(visit), "full", seed, rows, base, stock, held)


def scherer(args, ds, ret):
    with np.load(args.scherer) as z:
        people, session = z["subject"], z["session"]
    run = orders.runs_from_manifest(args.manifest, people, session)
    names = sorted(set(people))
    for scheme in ("full", "within"):
        for k in range(4):
            seed = 71000 + k
            for key, rows in orders.plans(people, session, run, scheme, seed).items():
                name, visit = key.split("|")
                if ret and visit != "2":
                    continue
                who = names.index(name)
                tag = f"{scheme}|{k}|{key}"
                with np.load(args.niv / "cap250" / f"p{who}_c{k // 2}.npz") as z:
                    new = read_labels(z, tag, rows, new=True)
                if ret:
                    with np.load(args.niv / f"ret_cap250_k{who}" / f"p{who}_c0.npz") as z:
                        stock = read_labels(z, tag, rows)
                else:
                    stock = new
                yield Stream("sch_" + name.split("/")[-1], int(visit), scheme, seed, rows, new, stock, new)


def netbci(args, ds, ret):
    for i in range(19):
        who = f"net_p{i}"
        with np.load(args.b10 / "ext" / f"p{i}.npz") as z:
            day, name, run = z["day"], str(z["name"]), z["run"]
        stream = np.array([f"{name}|{v}" for v in day])
        for scheme in ("full", "within"):
            for seed in range(72000, 72004):
                order = orders.permutation(stream, run, scheme, seed)
                for visit in (2, 3, 4):
                    rows = order[day[order] == visit]
                    prefix = f"{scheme}|{seed}|{visit}"
                    def labels(suffix):
                        path = args.b10 / "out" / f"v66_cp1_cap250{suffix}_k{i}" / "ret.npz"
                        with np.load(path) as z:
                            return read_labels(z, prefix, rows, new=bool(suffix))
                    new = labels("_new")
                    yield Stream(who, visit, scheme, seed, rows, new, labels("") if ret else new, new)
