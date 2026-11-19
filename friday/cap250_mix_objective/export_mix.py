"""Qualify each continued pair against original bounded CPU source replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, row_sha, sha
from export_checks import check, reference, reset
from held import frozen_member
from incremental_model import export_actor
from replay_f13 import DATA, origin_report


HERE = Path(__file__).resolve().parent
ARMS = ("hard","nominal","matched")
PUBLISHED = ("ec33684850e902479f3bf28e472a9ea590e64fa759b0b03704daf7aff1ea4018",
             "ed83b6b5b2a2765f82afa4a2fcb3f2c526d24a32e8b789b527268bde67b5401e")


def receipt(path,**record):
    with path.open("x") as f:
        json.dump(record,f,sort_keys=True,allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    return sha(path)


def export(seed,smoke):
    assert str(torch.__version__) == "2.14.1+cu130"
    torch.set_num_threads(1)
    origin_report()
    assert sha(DATA) == DATA_SHA
    with np.load(DATA,allow_pickle=False) as d:
        x,people,visit = (d[k] for k in ("X","subject","session"))
    sessions = np.array([f"{p}|{s}" for p,s in zip(people,visit,strict=True)])
    choices = sorted(set(sessions))
    assert len(choices)==14 and len(set(people))==10 and x.shape==(2655,41,480)
    picked = [np.flatnonzero(sessions==k)[:48] for k in (choices[0],choices[-1])]
    sample = [np.ascontiguousarray(x[r]) for r in picked]
    assert all(a.shape==(48,41,480) for a in sample)
    source = sha(HERE/"fit.sha256")
    manifest = sha(HERE/"export.sha256")
    smoke_fit = json.loads((HERE/"fit-smoke.json").read_text())
    assert smoke_fit["passed"] and not smoke_fit["held_labels_opened"] and smoke_fit["source"]==source
    dest = HERE/("smoke_exports" if smoke else "exports")
    dest.mkdir(exist_ok=True)
    if not smoke:
        validated = json.loads((HERE/"smoke_exports/s0.json").read_text())
        assert validated["source"]==source and validated["exporter"]==manifest and not validated["held_labels_opened"]
    results = {}
    for kind in (("matched",) if smoke else ARMS):
        path = HERE/"checkpoints"/f"{kind}_{seed}_all_source.pt"
        proof = path.with_suffix(".json")
        m = json.loads(proof.read_text())
        found = dict(held="all_source",kind=kind,seed=seed,checkpoint=sha(path),data=DATA_SHA,source=source,
                     steps=112,held_labels_opened=False,source_only_labels=True,normalized_population_immutable=True)
        if any(m[k]!=v for k,v in found.items()) or any(m[k]!=v for k,v in smoke_fit["bindings"].items()):
            raise ValueError("actual source continuation receipt differs")
        if (path.stat().st_mode | proof.stat().st_mode) & 0o222 or m["state"]==m["initial"] or not 0<m["fitted_movement"]:
            raise ValueError("continued source weights not immutable or changed")
        native,layer = frozen_member(path,m)
        target,active = reference(native,layer,sample,m["scale"])
        assert active > 0
        actor = export_actor(native,layer,m["scale"],torch.from_numpy(sample[0][:1]))
        with torch.jit.optimized_execution(False):
            reset(actor)
        folder = dest/kind
        folder.mkdir(exist_ok=True)
        saved = folder/f"raw_s1c_s{seed}.pt"
        pending = folder/f".{saved.name}.pending"
        if pending.exists() or saved.exists():
            raise FileExistsError(saved)
        torch.jit.save(actor,str(pending))
        measured = check(torch.jit.load(str(pending),map_location="cpu").eval(),sample,target)
        pending.rename(saved)
        saved.chmod(0o444)
        results[kind] = dict(measured,path=str(saved),sha256=sha(saved),state=m["state"],checkpoint=sha(path),
                             source_receipt=sha(proof),bounded_active=active)
    digest = receipt(dest/f"s{seed}.json",seed=seed,smoke=smoke,exporter=manifest,source=source,
                     source_tape=smoke_fit["bindings"]["tape"],source_only=True,data=DATA_SHA,
                     held_labels_opened=False,probe_rows=[row_sha(a) for a in picked],results=results)
    print("PASS F-C31e CPU export",seed,smoke,digest,flush=True)


def seal():
    source = sha(HERE/"fit.sha256")
    manifest = sha(HERE/"export.sha256")
    pins = {k:[] for k in ARMS}
    bindings = {}
    inputs = []
    for seed in (0,1):
        path = HERE/"exports"/f"s{seed}.json"
        m = json.loads(path.read_text())
        if (m["seed"] != seed or m["smoke"] or m["held_labels_opened"] or not m["source_only"]
                or m["exporter"] != manifest or m["source"] != source or set(m["results"]) != set(ARMS)):
            raise ValueError("expected actual label-blind CPU proof and complete source pair")
        inputs.append(m["source_tape"])
        states = set()
        for kind,a in m["results"].items():
            raw = Path(a["path"])
            prior = HERE/"checkpoints"/f"{kind}_{seed}_all_source.pt"
            if (sha(raw)!=a["sha256"] or sha(prior)!=a["checkpoint"] or sha(prior.with_suffix(".json"))!=a["source_receipt"]
                    or a["windows"]!=96 or a["max_abs"]>2e-5 or a["changed_decisions"] or a["bounded_active"]<=0
                    or any(a[x]!=2 for x in ("prefix","first_windows","reset"))):
                raise ValueError("CPU artifact hash or actual bounded parity differs")
            states.add(a["state"])
            pins[kind].append(sha(raw))
        if len(states) != len(ARMS):
            raise ValueError("complete trained state duplication")
        bindings[path.name] = sha(path)
    if len(set(inputs)) != 1:
        raise ValueError("complete actors have different source tapes")
    digest = receipt(HERE/"actors.json",c31e_source_only=True,actors=dict(pins,hard_h100=list(PUBLISHED)),
                     source=source,exporter=manifest,bindings=bindings,source_tape=inputs[0],held_labels_opened=False)
    print("SEALED F-C31e exact CPU pairs",digest,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("smoke","export","seal"))
    p.add_argument("seed",type=int,choices=(0,1),nargs="?",default=0)
    a = p.parse_args()
    if a.mode == "seal":
        seal()
    else:
        export(a.seed,a.mode=="smoke")
