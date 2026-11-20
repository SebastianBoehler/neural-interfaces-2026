"""Use the Lane Z parent harness for new mixer arms in both shuffled deliveries."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from bridge_orders import write_npz, write_receipt
from bridge_owner import pin, sha, stamp
from complete_zero import FEAT, HASHES
from mixes import ITEMS, features, fit, learned, mix_hook, portfolio
from z_parent import B1REF, C20, F32, PUPIL_SOURCE, helper, parents, released_control, run_chain, session, tail

WORK = Path(__file__).resolve().parent
OTHER = {"full0": "within0", "within0": "full0"}


def load_inputs(order, cb_glob):
    pin(FEAT, HASHES[FEAT])
    with np.load(FEAT, allow_pickle=False) as z:
        people, person = np.unique([p.rsplit("/", 1)[-1] for p in z["subject"]], return_inverse=True)
        y, visits = z["y"].astype(int), z["session"]
        cells = np.array(["|".join(map(str, row)) for row in zip(person, visits, z["interface"], y, strict=True)])
    assert len(people) == 10 and len(y) == 2655
    parents(order)
    pupils, _ = helper("lock_f32_inputs", F32 / "consume32.py").inputs(order, people[person], PUPIL_SOURCE)
    seeds = sorted(cb_glob.parent.glob(cb_glob.name))
    assert len(seeds) == 3
    cb, delivery = [], None
    for file in seeds:
        with np.load(file, allow_pickle=False) as z:
            assert str(z["order"]) == order
            if delivery is not None:
                np.testing.assert_array_equal(delivery, z["delivery"])
            delivery = z["delivery"]
            cb.append(z["post"])
    return person, visits, y, cells, .8 * pupils["hard"] + .2 * np.mean(cb, axis=0), delivery


def arms(order, person, y, cells):
    arm = {
        "stock_base": dict(system="stock"),
        "hardcb_base": dict(system="hardcb"),
        "stock_portfolio": dict(system="stock", mix_fn=mix_hook(lambda p, _: portfolio(p, ITEMS))),
        "hardcb_portfolio": dict(system="hardcb", mix_fn=mix_hook(lambda p, _: portfolio(p, ITEMS))),
    }
    with np.load(WORK / f"{OTHER[order]}_prof.npz", allow_pickle=False) as source:
        members = {k: source[f"hardcb|base|member|{k}"].astype(np.float64) for k, _ in ITEMS}
    for name, interaction in (("additive", False), ("stack", True)):
        logp, x = features(members, ITEMS, interaction)
        coef = []
        for held in range(10):
            train = person != held
            coef.append(fit(logp[train], x[train], y[train], cells[train]))
        arm[f"hardcb_{name}"] = dict(system="hardcb", mix_fn=mix_hook(learned(np.stack(coef), person, interaction)))
    return arm


def run(order, smoke, cb_glob):
    if not smoke:
        proof = json.loads((WORK / f"{order}_smoke.json").read_text())
        assert proof["passed"] and proof["source"] == sha(WORK / "source.sha256")
    person, visits, y, cells, pupil, delivery = load_inputs(order, cb_glob)
    plan = arms(order, person, y, cells)
    control = released_control(order)
    tm = helper("lock_parent_tail", C20 / "worker.py")
    out, times = {}, {}

    def runner(ctx, archive, patches, incumbent, stock):
        np.testing.assert_array_equal(stock, control["raw"])
        for name, spec in plan.items():
            start = perf_counter()
            raw, taps = run_chain(ctx, archive, patches, incumbent, spec, pupil, None, None, None)
            out[name], times[name] = (raw, taps["mix"]), {"chain": perf_counter() - start}
            print(f"{order} {name} live chain", flush=True)

    with session(order, runner) as (ctx, baseline):
        np.testing.assert_array_equal(ctx["delivery"], delivery)
        np.testing.assert_array_equal(np.sort(delivery), np.arange(len(y)))
        with (np.load(WORK / f"{order}_prof.npz", allow_pickle=False) as reference,
              np.load(B1REF / f"{order}_S.npz", allow_pickle=False) as hardcb):
            np.testing.assert_array_equal(delivery, reference["delivery"])
            count = ctx["count"]
            streams = tuple(count.SESS)
            rows = {s: count.C.order_rows(order, s, count.PROTO) for s in streams}
            assert len(streams) == 14
            if smoke:
                streams = tuple(s for s in streams if (person[rows[s]] == 0).all() and (visits[rows[s]] == "1").all())
                assert len(streams) == 1
            selected = np.concatenate([rows[s] for s in streams])
            arrays = {"delivery": delivery}
            for name, (raw, mix) in out.items():
                start = perf_counter()
                filtered, terminal = tail(ctx, order, raw, streams, set(), tm)
                times[name]["tail"] = perf_counter() - start
                system = plan[name]["system"]
                if name.endswith("_base"):
                    for stage, value in (("tsem", raw), ("mix", mix), ("filtered", filtered), ("terminal", terminal)):
                        np.testing.assert_array_equal(value[selected].astype(np.float32),
                                                      reference[f"{system}|base|{stage}"][selected])
                    for stage, value in (("raw", raw), ("filtered", filtered), ("terminal", terminal)):
                        expected = baseline[stage] if system == "stock" else hardcb[f"hardcb|{stage}"]
                        np.testing.assert_array_equal(value[selected], expected[selected])
                else:
                    assert np.count_nonzero(mix[selected] != out[f"{system}_base"][1][selected])
                    assert np.count_nonzero(terminal[selected] != arrays[f"{system}_base|terminal"][selected])
                arrays[f"{name}|mix"] = mix
                arrays[f"{name}|terminal"] = terminal
    stem = f"{order}_{'smoke' if smoke else 'complete'}"
    receipt = dict(passed=True, job=os.environ["SLURM_JOB_ID"], order=order, rows=len(selected),
                   source=sha(WORK / "source.sha256"), delivery=stamp(delivery), times=times,
                   cb_pupils=[sha(f) for f in sorted(cb_glob.parent.glob(cb_glob.name))],
                   label_use="other order/other person, meta-fit only",
                   profile_receipt=json.loads((WORK / f"{order}_prof.json").read_text())["source"])
    if not smoke:
        receipt["output"] = write_npz(WORK / f"{stem}.npz", arrays)
    write_receipt(WORK / f"{stem}.json", **receipt)
    print(f"PASS {stem} {len(selected)} rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order", choices=tuple(OTHER))
    parser.add_argument("mode", choices=("smoke", "complete"))
    parser.add_argument("cb_glob", type=Path)
    args = parser.parse_args()
    assert os.environ["Z_JOB"] == "lock"
    run(args.order, args.mode == "smoke", args.cb_glob)
