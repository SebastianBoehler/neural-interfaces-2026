"""Report the complete same-device F35 training-only comparison after sealing all decisions."""
import json

import numpy as np

from anchors import DATA_SHA,ORIGIN_SHA,PEOPLE,sha
from collect_covariance import CLASSES,class_recall,order_schemes,paired_summary
from contributions import ARMS,HERE,MODES
from tape_checks import ACTOR,F33,paired
from recipe import GPU
from replay_f13 import DATA,ORDER_SHA,ORDERS,ROOT,origin_report
from source_targets import TOL


COMPARISONS=(("balance","hard"),("balance","mirrored"),("mirrored","hard"))

def blind(subject,visit,person,streams):
    values=np.full((2,len(ARMS),len(ORDERS),len(subject)),-1,dtype=np.int8)
    files,diagnostic=[],{}
    first=resets=clips=checks=0
    drift=0.
    for seed in (0,1):
        diag={}
        for held in PEOPLE:
            path=HERE/"folds"/f"{held}_s{seed}.npz"
            digest=sha(path)
            if digest!=path.with_suffix(".sha256").read_text().strip():
                raise ValueError("A100 blind archive differs")
            rows=np.flatnonzero(person==held)
            n=len(set(streams[rows]))*len(ORDERS)
            fixed=dict(held=held,seed=seed,actor=ACTOR,scorer=sha(HERE/"score.sha256"),data=DATA_SHA,origin=ORIGIN_SHA,
                       orders=list(ORDERS),order_sha=ORDER_SHA,device=GPU,held_labels_opened=False)
            group=paired(held,seed,False,subject,visit)
            fields=("actual_original","actual_applied","actual_relative_original","actual_relative_applied","eligible")
            diag[held]={a:{k:m[k] for k in fields} for a,(_,m) in group.items() if a!="hard"}
            diag[held]["hard"]={k:group["hard"][1][k] for k in
                                ("anchor_relative_reduction","hard_gradient_error","eligible")}
            with np.load(path,allow_pickle=False) as z, np.load(F33/"folds"/path.name,allow_pickle=False) as prior:
                r=json.loads(str(z["metadata"]))
                if (any(r[k]!=v for k,v in fixed.items()) or not np.array_equal(z["index"],rows) or
                        set(z.files)!=set(ARMS)|{"index","metadata"} or set(r["evidence"])!=set(group)):
                    raise ValueError("A100 frozen row or receipt scope differs")
                if not np.array_equal(prior["index"],rows) or any(not np.array_equal(
                        z[f"hard_{m}"],prior[f"hard_{m}"]) for m in MODES):
                    raise ValueError("independent A100 hard decisions differ")
                bank=np.stack([z[a] for a in ARMS])
            if bank.shape!=(len(ARMS),len(ORDERS),len(rows)) or not np.isin(bank,(0,1,2)).all():
                raise ValueError("invalid A100 held decisions")
            for kind,(cp,meta) in group.items():
                v=r["evidence"][kind]
                expected=dict(checkpoint=meta["checkpoint"],state=meta["state"],
                              receipt=sha(cp.with_suffix(".json")),first_windows_exact=n,resets=2*n)
                if any(v[k]!=w for k,w in expected.items()) or not 0<=v["prefix_max_abs"]<=TOL or v["clips"]<=0:
                    raise ValueError("held prefix, bounded work or source fingerprint differs")
                first+=n
                resets+=2*n
                clips+=v["clips"]
                drift=max(drift,v["prefix_max_abs"])
            values[seed][:,:,rows]=bank
            checks+=len(MODES)*len(ORDERS)*len(rows)
            files.append(f"{digest}  folds/{path.name}\n")
        diagnostic[str(seed)]=diag
    if (values<0).any() or checks!=84_960:
        raise ValueError("full A100 matched decision bank is incomplete")
    lock=HERE/"before_labels.sha256"
    with lock.open("x") as out:
        out.writelines(sorted(files))
    return values,sha(lock),dict(first=first,resets=resets,active_clips=clips,prefix_max_abs=drift,hard_exact=checks),diagnostic


def statistics(values,person,streams,dataset):
    import protocol
    import score_s_v66p

    if (protocol.__file__!=str(ROOT/"experiments/protocol.py") or
            score_s_v66p.__file__!=str(ROOT/"studies/14nov/newmember/score_s_v66p.py")):
        raise ValueError("official held metrics ancestry differs")
    labels=dataset["y"]
    cells=np.char.add(np.char.add(streams,"|"),dataset["interface"])
    counts=np.asarray([len(np.unique(cells[person==p])) for p in PEOPLE])
    if len(labels)!=2655 or int(counts.sum())!=28:
        raise ValueError("gold rows or official held cells differ")
    schemes=order_schemes()
    results={}
    for seed,bank in enumerate(values):
        score,recall,across,paired_result,class_paired={},{},{},{},{}
        for arm,decisions in zip(ARMS,bank,strict=True):
            by_class=np.stack([class_recall(p,labels,cells,person) for p in decisions])
            people=np.stack([score_s_v66p.person_accuracy(np.eye(3)[p],labels,cells,person,PEOPLE) for p in decisions])
            orders=np.asarray([100*protocol.score(labels,p,dataset,np.ones(len(labels),bool))[
                "balanced_accuracy_cell_mean"] for p in decisions])
            np.testing.assert_allclose(by_class.mean(-1),people,atol=1e-10,rtol=0)
            np.testing.assert_allclose(np.average(people,weights=counts,axis=1),orders,atol=1e-10,rtol=0)
            across[arm]=by_class
            score[arm]=float(np.mean([orders[g].mean() for g in schemes]))
            class_orders=np.average(by_class,weights=counts,axis=1)
            recall[arm]=dict(zip(CLASSES,np.mean([class_orders[g].mean(0) for g in schemes],0).tolist()))
        for mode in MODES:
            for a,b in COMPARISONS:
                name=f"{mode}_{a}_minus_{b}"
                delta=across[f"{a}_{mode}"]-across[f"{b}_{mode}"]
                def summary(d):
                    return paired_summary(d,schemes,counts,score_s_v66p)
                paired_result[name]=summary(delta.mean(-1))
                class_paired[name]={c:summary(delta[:,:,i]) for i,c in enumerate(CLASSES)}
                class_paired[name]["cognitive_mean"]=summary(delta[:,:,1:].mean(-1))
        results[str(seed)]=dict(absolute=score,class_recall=recall,paired=paired_result,class_paired=class_paired)
    outcome={}
    for ref in ("hard","mirrored"):
        k=f"bounded_balance_minus_{ref}"
        a,b=(results[str(seed)]["paired"][k]["interval"] for seed in (0,1))
        outcome[k]="supported" if min(a[0],b[0])>0 else "refuted" if max(a[1],b[1])<=0 else "undecided"
    return results,outcome


def run():
    import protocol

    origin_report()
    with np.load(DATA,allow_pickle=False) as z:
        subject,visit=z["subject"],z["session"]
    person=np.asarray([s.rsplit("/",1)[-1] for s in subject])
    streams=np.char.add(np.char.add(subject,"|"),visit)
    values,lock,guards,diag=blind(subject,visit,person,streams)
    d=protocol.load()
    if not np.array_equal(d["subject"],subject) or not np.array_equal(d["stream"],streams):
        raise ValueError("gold identities differ from locked A100 predictions")
    scores,outcome=statistics(values,person,streams,d)
    diagnostics={}
    for seed,by_person in diag.items():
        mass=np.asarray([z["hard"]["eligible"] for z in by_person.values()])
        mean={role:{k:float(np.average([z[role][k] for z in by_person.values()],weights=mass))
                    for k in ("actual_relative_original","actual_relative_applied")}
              for role in ("balance","mirrored")}
        reality=1-mean["balance"]["actual_relative_applied"]/mean["mirrored"]["actual_relative_applied"]
        diagnostics[seed]=dict(folds=by_person,mean=mean,balance_vs_mirrored_relative_reduction=reality,
                               diagnostic_prediction_met=reality>=.05)
    text=json.dumps(dict(scope="held-new WHOLE DEV",device=GPU,actor=ACTOR,scorer=sha(HERE/"score.sha256"),
                         before_labels=lock,guards=guards,source_contributions=diagnostics,
                         seeds=scores,outcome=outcome),sort_keys=True,indent=2,allow_nan=False)
    path=HERE/"whole_results.json"
    with path.open("x") as out:
        out.write(text+"\n")
    print(f"PASS F35 whole score {sha(path)}",flush=True)


if __name__=="__main__":
    run()
