"""Seal F33 blind held decisions and report the original whole-decoder metric."""
import json

import numpy as np

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, sha
from held_covariance import ACTOR, F16, paired
from covariance import ARMS, HERE, KINDS, MODES
from replay_f13 import DATA, ORDER_SHA, ORDERS, ROOT, origin_report
from source_targets import TOL


F16_RESULT = "e91c377bd9560d7c02f01adc9ccaa5e82b87d347927c19e17fd1d217780a803e"
COMPARISONS = (("aligned", "hard"), ("aligned", "permuted"), ("permuted", "hard"))


def lock_decisions(subject, visit, person, streams):
    values = np.full((2, len(ARMS), len(ORDERS), len(subject)), -1, dtype=np.int8)
    manifest = []
    clips = windows = resets = 0
    prefix = 0.
    for seed in (0, 1):
        for held in PEOPLE:
            path = HERE / "folds" / f"{held}_s{seed}.npz"
            digest = sha(path)
            if digest != path.with_suffix(".sha256").read_text().strip():
                raise ValueError("held decisions changed")
            manifest.append(f"{digest}  folds/{path.name}\n")
            rows = np.flatnonzero(person == held)
            n = len(set(streams[rows])) * len(ORDERS)
            required = dict(held=held, seed=seed, actor=ACTOR, scorer=sha(HERE / "score.sha256"), data=DATA_SHA,
                            origin=ORIGIN_SHA, orders=list(ORDERS), order_sha=ORDER_SHA, held_labels_opened=False)
            with np.load(path, allow_pickle=False) as z, np.load(F16 / "folds" / path.name, allow_pickle=False) as prior:
                receipt = json.loads(str(z["metadata"]))
                if (any(receipt[k] != v for k, v in required.items()) or not np.array_equal(z["index"], rows) or
                        set(z.files) != set(ARMS) | {"metadata", "index"} or set(receipt["evidence"]) != set(KINDS) or
                        not np.array_equal(prior["index"], rows) or any(
                            not np.array_equal(z[f"hard_{m}"], prior[f"hard_{m}"]) for m in MODES)):
                    raise ValueError("blind F33 rows, original hard control or provenance differs")
                bank = np.stack([z[a] for a in ARMS])
            if bank.shape != (len(ARMS), len(ORDERS), len(rows)) or not np.isin(bank, (0, 1, 2)).all():
                raise ValueError("invalid held decision bank")
            for kind, (cp, source) in paired(held, seed, False, subject, visit).items():
                proof = receipt["evidence"][kind]
                expected = dict(checkpoint=source["checkpoint"], state=source["state"], permutation=source.get("permutation"),
                                receipt=sha(cp.with_suffix(".json")), first_windows_exact=n, resets=2 * n)
                if any(proof[k] != v for k, v in expected.items()) or not 0 <= proof["prefix_max_abs"] <= TOL:
                    raise ValueError("held archive or prefix controls differ")
                clips += proof["clips"]
                windows += n
                resets += 2 * n
                prefix = max(prefix, proof["prefix_max_abs"])
            values[seed][:, :, rows] = bank
    if (values < 0).any() or not clips:
        raise ValueError("full held matrix or bounded work missing")
    lock = HERE / "before_labels.sha256"
    with lock.open("x") as out:
        out.writelines(sorted(manifest))
    return values, sha(lock), dict(first=windows, resets=resets, active_clips=clips, prefix_max_abs=prefix)


def order_schemes():
    groups = [np.flatnonzero([order.startswith(s) for order in ORDERS])
              for s in ("within", "full", "runs")]
    if [len(g) for g in groups] != [4, 2, 2] or not np.array_equal(
            np.sort(np.concatenate(groups)), np.arange(len(ORDERS))):
        raise ValueError("pinned order groups are incomplete or overlap")
    return groups


CLASSES = ("motor", "subtraction", "word")


def class_recall(decision, labels, cells, person):
    table = np.empty((len(PEOPLE),len(CLASSES)),dtype=np.float64)
    for i, name in enumerate(PEOPLE):
        local = person == name
        rates=[]
        for cell in np.unique(cells[local]):
            mask = local & (cells == cell)
            by_class = [mask & (labels == c) for c in range(len(CLASSES))]
            if any(not rows.any() for rows in by_class):
                raise ValueError("missing class in a scored expert cell")
            rates.append([100 * np.mean(decision[rows] == c) for c,rows in enumerate(by_class)])
        table[i]=np.mean(rates,axis=0)
    return table


def paired_summary(delta, by_scheme, counts, metric):
    individuals=np.mean([delta[g].mean(0) for g in by_scheme],0)
    return dict(**metric.cell_weighted_summary(delta,ORDERS,counts),
                people=dict(zip(PEOPLE,individuals.tolist())),
                orders=dict(zip(ORDERS,np.average(delta,weights=counts,axis=1).tolist())))


def statistics(values, person, streams, dataset, by_scheme):
    import protocol
    import score_s_v66p

    if (protocol.__file__ != str(ROOT / "experiments/protocol.py") or
            score_s_v66p.__file__ != str(ROOT / "studies/14nov/newmember/score_s_v66p.py")):
        raise ValueError("whole-decoder metric ancestry differs")
    labels=dataset["y"]
    cells=np.char.add(np.char.add(streams,"|"),dataset["interface"])
    counts=np.asarray([len(np.unique(cells[person == p])) for p in PEOPLE])
    if len(labels)!=2655 or int(counts.sum())!=28:
        raise ValueError("held labels or official cells differ")
    ref=F16/"whole_results.json"
    if sha(ref)!=F16_RESULT:
        raise ValueError("F16 hard control results differ")
    original=json.loads(ref.read_text())["seeds"]
    results={}
    for seed,bank in enumerate(values):
        accuracy, across, recalls, class_across, paired_result, class_paired = {}, {}, {}, {}, {}, {}
        for arm,decision in zip(ARMS,bank,strict=True):
            across[arm]=np.stack([score_s_v66p.person_accuracy(np.eye(3)[p],labels,cells,person,PEOPLE) for p in decision])
            class_across[arm]=np.stack([class_recall(p,labels,cells,person) for p in decision])
            official=np.asarray([100 * protocol.score(labels,p,dataset,np.ones(len(labels),bool))[
                "balanced_accuracy_cell_mean"] for p in decision])
            np.testing.assert_allclose(np.average(across[arm],weights=counts,axis=1),official,atol=1e-10,rtol=0)
            np.testing.assert_allclose(class_across[arm].mean(axis=2),across[arm],atol=1e-10,rtol=0)
            accuracy[arm]=float(np.mean([official[g].mean() for g in by_scheme]))
            class_orders=np.average(class_across[arm],weights=counts,axis=1)
            recalls[arm]=dict(zip(CLASSES,np.mean([class_orders[g].mean(0) for g in by_scheme],0).tolist()))
        for mode in MODES:
            key=f"hard_{mode}"
            if abs(accuracy[key]-original[str(seed)]["absolute"][key])>1e-10:
                raise ValueError("F16 sealed numeric hard score changed")
            for a,b in COMPARISONS:
                first,second=f"{a}_{mode}",f"{b}_{mode}"
                key=f"{mode}_{a}_minus_{b}"
                paired_result[key]=paired_summary(across[first]-across[second],by_scheme,counts,score_s_v66p)
                delta=class_across[first]-class_across[second]
                class_paired[key]={name:paired_summary(delta[:,:,c],by_scheme,counts,score_s_v66p)
                                   for c,name in enumerate(CLASSES)}
                class_paired[key]["cognitive_mean"]=paired_summary(delta[:,:,1:].mean(axis=2),by_scheme,counts,score_s_v66p)
        results[str(seed)]=dict(absolute=accuracy,paired=paired_result,class_recall=recalls,class_paired=class_paired)
    outcomes={}
    for control in ("hard","permuted"):
        key=f"bounded_aligned_minus_{control}"
        a,b=(results[str(seed)]["paired"][key]["interval"] for seed in (0,1))
        outcomes[key]="supported" if min(a[0],b[0])>0 else "refuted" if max(a[1],b[1])<=0 else "undecided"
    return results,outcomes



def run():
    import protocol

    by_scheme = order_schemes()
    origin_report()
    with np.load(DATA, allow_pickle=False) as z:
        subject, visit = z["subject"], z["session"]
    person = np.asarray([s.rsplit("/", 1)[-1] for s in subject])
    streams = np.char.add(np.char.add(subject, "|"), visit)
    values, seal, guards = lock_decisions(subject, visit, person, streams)
    d = protocol.load()
    if not np.array_equal(d["subject"], subject) or not np.array_equal(d["stream"], streams):
        raise ValueError("gold rows differ from sealed held decisions")
    diagnostic = {}
    for seed in (0, 1):
        folds = {}
        for held in PEOPLE:
            checked = paired(held, seed, False, subject, visit)
            keys=("left_gradient","right_gradient","relation_gradient","gain_error","energy_error",
              "feature_delta","logit_delta","residual")
            folds[held]={k:{field:m[field] for field in keys} for k,(_,m) in checked.items() if k!="hard"}
        means={k:{f:float(np.mean([v[k][f] for v in folds.values()])) for f in keys} for k in ("aligned","permuted")}
        diagnostic[str(seed)] = dict(folds=folds, means=means)
    scores, outcome = statistics(values, person, streams, d, by_scheme)
    scorer = sha(HERE / "score.sha256")
    result = dict(scope="held-new WHOLE DEV", actor=ACTOR, scorer=scorer,
                  before_labels=seal, f16=F16_RESULT, guards=guards, source_covariance=diagnostic, seeds=scores, outcome=outcome)
    rendered = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    path = HERE / "whole_results.json"
    with path.open("x") as out:
        out.write(rendered + "\n")
    print(f"PASS F33 whole score {sha(path)}", flush=True)


if __name__ == "__main__":
    run()
