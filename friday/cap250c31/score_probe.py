"""Authenticate all causal probe decisions, then open gold for the official cell metric."""
import json
from pathlib import Path

import numpy as np

from locked import ARMS, NOMINAL, PARENT_SEAL, SEEDS, parent, parent_seal, pin, scope, select, sha, stamp, write
from protocol import score


def qualified():
    who, day, interface, _, _, plans = scope()
    seal = parent_seal()
    source = sha("source.sha256")
    smoke = json.loads(Path("smoke.json").read_text())
    assert smoke["passed"] and smoke["source"] == source and smoke["parents"] == PARENT_SEAL
    assert smoke["state_exact"] and smoke["reset_prefix_exact"] and not smoke["held_labels_opened"]
    people = sorted(set(who))
    shape = len(SEEDS), len(who)
    bank = np.full((len(ARMS), *shape), -1, np.int8)
    eligibility = np.zeros(shape, bool)
    ages = np.full(shape, -1, int)
    activity = dict(posterior_peak=0., actor_label_changes=0)
    signed = {}
    for index, person in enumerate(people):
        receipt, path = Path(f"bank_{index}.json"), Path(f"bank_{index}.npz")
        assert not (receipt.stat().st_mode | path.stat().st_mode) & 0o222
        r = json.loads(receipt.read_text())
        assert r["passed"] and r["who"] == person and r["index"] == index and r["state_exact"]
        assert r["source"] == source and r["smoke"] == sha("smoke.json") and r["parents"] == PARENT_SEAL
        assert r["held_labels_opened"] is False
        pin(path, r["bank"])
        parents = parent(index, person, plans, seal)
        assert r["keys"] == sorted(parents)
        signed[receipt.name], signed[path.name] = sha(receipt), r["bank"]
        with np.load(path, allow_pickle=False) as probe:
            for key, (rows, labels, _) in parents.items():
                k = SEEDS.index(int(key.split("|")[-1]))
                np.testing.assert_array_equal(rows, probe[key + "|rows"])
                assert (bank[:, k, rows] == -1).all()
                p = probe[key + "|actors"]
                assert p.shape == (len(rows), 2, 2, 3) and p.dtype == np.float64 and (p >= 0).all()
                np.testing.assert_allclose(p.sum(-1), 1, atol=1e-12, rtol=0)
                peak = float(np.max(np.abs(p[:, 0] - p[:, 1])))
                assert peak > 1e-3
                activity["posterior_peak"] = max(activity["posterior_peak"], peak)
                activity["actor_label_changes"] += int(np.count_nonzero(p.mean(2).argmax(2)[:, 0] != p.mean(2).argmax(2)[:, 1]))
                lq = np.log(np.maximum(p.mean(2), 1e-12))
                expected, present = select(lq[:, 1] - lq[:, 0], labels)
                np.testing.assert_array_equal(expected, probe[key + "|labels"])
                np.testing.assert_array_equal(present, probe[key + "|eligible"])
                bank[:, k, rows] = expected
                eligibility[k, rows] = present
                ages[k, rows] = np.arange(len(rows))
    assert np.isin(bank, (0, 1, 2)).all() and (ages >= 0).all()
    pin_parent = stamp(bank[:3])
    assert pin_parent == seal["predictions"]
    assert np.count_nonzero(bank[0] != bank[2]) == 75
    assert (eligibility == ((bank[0] != bank[2]) & (ages > 0))).all()
    assert np.array_equal(bank[2], bank[3]) and (bank[4:][:, ~eligibility] == bank[2][~eligibility]).all()
    signature = write("before_labels.json", dict(held_labels_opened=False, source=source,
        scoring=sha("scoring.sha256"), smoke=sha("smoke.json"), parents=PARENT_SEAL, old_decisions=pin_parent,
        decisions=stamp(bank), eligibility=stamp(eligibility), age=stamp(ages), signed=signed))
    return bank, eligibility, ages, activity, who, day, interface, people, signature


def official(bank, who, day, interface, gold, people):
    cells = np.empty((len(bank), len(people), 2, len(SEEDS), 3), float)
    for i, person in enumerate(people):
        for j, visit in enumerate(("1", "2")):
            for c in range(3):
                take = (who == person) & (day == visit) & (gold == c)
                assert take.any()
                cells[:, i, j, :, c] = 100 * np.mean(bank[:, :, take] == c, axis=-1)
    means = cells.mean(-1)
    stream = np.char.add(np.char.add(who, "|"), day)
    for a in range(len(bank)):
        for k in range(len(SEEDS)):
            ref = score(gold, bank[a, k], dict(stream=stream, interface=interface), np.ones(len(who), bool))
            assert ref["n_cells"] == 18
            np.testing.assert_allclose(np.sort(100 * np.asarray(ref["cell_scores"])),
                                       np.sort(means[a, :, :, k].ravel()), rtol=0, atol=1e-10)
    return means, cells


def bootstrap():
    rng = np.random.default_rng(625)
    i = rng.integers(9, size=(14000, 9, 1, 1))
    v = rng.integers(2, size=(14000, 9, 2, 1))
    o = rng.integers(2, size=(14000, 9, 2, 2))

    def summary(delta):
        return dict(pp=float(delta.mean()), paired80=np.percentile(delta[i, v, o].mean((1, 2, 3)), [10, 90]).tolist(),
                    order_pp=delta.mean((0, 1)).tolist(), person_pp=delta.mean((1, 2)).tolist())

    return summary


def run():
    bank, present, ages, activity, who, day, interface, people, signature = qualified()
    with np.load(NOMINAL, allow_pickle=False) as data:
        gold = data["y"]
    stock, repeat, hard, _, sham, real = bank
    switched = real != hard
    rescue = (stock == gold) & (hard != gold)
    options = (np.where(switched & rescue, stock, hard), np.where(present & rescue, stock, hard),
               np.where(rescue, stock, hard))
    arms = ARMS + ("current_flags_oracle", "age_eligible_oracle", "all_disagreements_oracle")
    means, classes = official(np.concatenate((bank, options)), who, day, interface, gold, people)
    a, b, h, _, sh, tr, flag, eligible, all_rows = means
    summarize = bootstrap()
    contrasts = dict(current_minus_hard=summarize(tr - h), current_minus_sham=summarize(tr - sh),
        current_minus_stock=summarize(tr - (a + b) / 2), sham_minus_hard=summarize(sh - h),
        flags_oracle_minus_hard=summarize(flag - h), eligible_oracle_minus_hard=summarize(eligible - h),
        all_oracle_minus_hard=summarize(all_rows - h))
    actual = []
    for k, seed in enumerate(SEEDS):
        def moves(out):
            changed = out[k] != hard[k]
            return dict(switches=int(changed.sum()), fixes=int((changed & rescue[k]).sum()),
                        losses=int((changed & (hard[k] == gold)).sum()))
        actual.append(dict(seed=seed, total_disagreements=int(np.count_nonzero(stock[k] != hard[k])),
            eligible=int(present[k].sum()), oracle_fixes=int(rescue[k].sum()),
            eligible_oracle_fixes=int((present[k] & rescue[k]).sum()),
            hard_boot_first=int(np.count_nonzero((stock[k] != hard[k]) & (ages[k]==0))), real=moves(real), sham=moves(sham)))
    primary = contrasts["current_minus_hard"]
    check = contrasts["current_minus_sham"]
    advance = primary["pp"] >= .30 and all(x["paired80"][0] > 0 and min(x["order_pp"]) > 0 for x in (primary, check))
    effect = classes[5] - classes[2]
    summary = dict(scope="FULL-SYSTEM EXTERNAL DEV retrospective decision selection", orders=SEEDS,
        parents=PARENT_SEAL, before_labels=signature, people=people, contrasts=contrasts, activity=activity,
        advances=bool(advance), refuted={k: v["paired80"][1] <= 0 for k, v in contrasts.items()
                                        if k.startswith("current_minus_")},
        class_current_minus_hard=[summarize(effect[..., c]) for c in range(3)],
        absolute_pct=dict(zip(arms, means.mean((1, 2, 3)).tolist(), strict=True)), switches=actual)
    print("PASS C31 exact parent decision comparison", write("result.json", summary), contrasts, actual, flush=True)


if __name__ == "__main__":
    run()
