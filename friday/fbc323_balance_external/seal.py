"""Authenticate both full source receipts and seal the six CPU exports."""
import json
from pathlib import Path

from anchors import sha
from all_source import BUNDLE,F35,HERE,scope,tag
from contributions import KINDS
from external_checks import ACTOR,authenticated_group
from recipe import GPU
from replay_f13 import DATA


def run():
    _,rows,streams,people,_=scope(DATA)
    files=[HERE/"actor.sha256",HERE/"export.sha256"]
    for seed in (0,1):
        receipt=HERE/"exports"/f"s{seed}.json"
        record=json.loads(receipt.read_text())
        expected=dict(seed=seed,smoke=False,device="cpu",fit_device=GPU,actor=ACTOR,exporter=sha(HERE/"export.sha256"),
                      f25e=BUNDLE,f35=F35,scherer_exposed=False,gold_opened=False)
        if any(record[k]!=v for k,v in expected.items()) or set(record["results"])!=set(KINDS):
            raise ValueError("full export receipt or scope differs")
        files.append(receipt)
        suffix=tag(seed,False)
        files.append(HERE/"checkpoints"/f"tape_{suffix}.npz")
        for kind,(source,meta) in authenticated_group(seed,False,rows,streams,people).items():
            result=record["results"][kind]
            actor=HERE/"exports"/kind/f"raw_s1c_s{seed}.pt"
            cert=dict(path=str(actor),sha256=sha(actor),state=meta["state"],checkpoint=sha(source),
                      source_receipt=sha(source.with_suffix(".json")),windows=96,changed_decisions=0,
                      first_windows=2,prefix=2,reset=2,tape=meta["tape"],trajectory=meta.get("trajectory"))
            if any(result[k]!=v for k,v in cert.items()) or not 0<=result["max_abs"]<=2e-5 or result["bounded_active"]<=0:
                raise ValueError("CPU export exceeds strict source-only proof")
            files.extend((actor,source,source.with_suffix(".json")))
            if kind!="hard":
                files.append(HERE/"checkpoints"/f"{kind}_trajectory_{suffix}.npz")
    path=HERE/"artifact.sha256"
    with path.open("x") as out:
        out.writelines(f"{sha(p)}  {p.relative_to(HERE)}\n" for p in sorted(files))
    print(f"PASS F35e signed CPU source bundle {sha(path)}",flush=True)


if __name__=="__main__":
    run()
