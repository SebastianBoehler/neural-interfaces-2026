"""Prove the exact hard package still consumes warmed raw evidence after age 250."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from exact import TRIAL, expected_cpu
from qualify import load, measured, proof, protect
import replay
from settings import sha, stamp

STRESS = "815138d6536adb999fdfdf37dfc243f66b3718b82bf8198ccd7cc04a1f1c53b5"
ROWS = "c5b80556953e87b39a223e78e578eb666364fd0906c5b9741268a40eac39f0c7"


def run(stock_zip,trial_zip):
    assert torch.__version__=="2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    assert sha(trial_zip)==TRIAL and sha("/reference.json")==STRESS
    torch.set_num_threads(1)
    raw,_,channels=replay.read_source()
    rows=np.random.default_rng(36017800).permutation(len(raw))[:378].astype("<i8")
    assert stamp(rows)==ROWS
    windows=raw[rows].to("cuda:0")
    expected=expected_cpu(raw,rows)
    reference=json.loads(Path("/reference.json").read_text())["streams"]["first"]["forced-unknown-378"]["activity"]
    with TemporaryDirectory() as root:
        stock,trial=load(stock_zip,trial_zip,Path(root),channels)
        for model in (stock,trial):
            assert model.ft.cap==250 and 0<=model.route.identity.tau<1e6 and 0<=model.route.contrast.tau<1e6
            for gate in (model.route.identity,model.route.contrast):
                gate.tau=1e6
        with torch.jit.optimized_execution(False):
            original=measured(stock,windows)
            changed=measured(trial,windows,expected)
        migration=protect(original,changed,None)
        active=proof(stock,changed[1],changed[3])
        model,view,refits,probe=changed
        assert model.ft.steps==480 and model.ft.capped==12 and model.ft.over==0
        for fresh in (original[0],model):
            assert all(len(k)==250 for k in fresh.ft.X.values())
            assert len(fresh.ft.lab)==len(fresh.ft.wt)==378
        assert model.ft.lab==original[0].ft.lab
        assert [r["age"] for r in refits]==list(range(20,251,10))
        assert active["calls"]==378 and active["warm_rows"]==358 and active["csp_fusions"]==288
        assert all("fbraw" in row[2] for row in probe.inputs[260:])
        source=deepcopy(stock)
        post=[]
        for age in range(260,378):
            m,members,warm,rest=probe.inputs[age]
            source.rest_p=rest
            old=source.mix6(m,members,warm)
            post.append(float(np.abs(old-view["events"][age][2]).max()))
        assert len(post)==118 and min(post)>0 and max(post)>.01
        for name, value in reference.items():
            if name in original[1]["work"]:
                assert original[1]["work"][name]==value, (name,original[1]["work"][name],value)
    result=dict(passed=True,held_labels_opened=False, scope="forced-unrecognized public cap stress",
                archive=sha(trial_zip),source=sha("source.sha256"),rows=ROWS,stock_work=original[1]["work"],
                hard_work=view["work"],archived_stock=reference,proof=active,post_cap_rows=len(post),
                post_cap_mix_min=min(post),post_cap_mix_max=max(post),independent_numeric=migration)
    name=Path("stress.json")
    with name.open("x") as fp:
        json.dump(result,fp,sort_keys=True,allow_nan=False)
        fp.write("\n")
    name.chmod(0o444)
    print("PASS F-C25pc",sha(name),result["hard_work"],result["post_cap_mix_max"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stock",type=Path)
    p.add_argument("trial",type=Path)
    a=p.parse_args()
    run(a.stock,a.trial)
