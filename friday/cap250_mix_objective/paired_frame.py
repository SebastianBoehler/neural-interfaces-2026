"""Pair natural and threshold-suppressed source sessions, without scoring."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

import replay
from source_frame import ARCHIVE, DATA, DELIVERY, inputs, private, sha, stamp, stream, write


STRESS = "95a1fc3d4bf3f0ddd2491f90b6181f945f3446b093c9814f65cc387deaacfe4a"
CANARY = "c4330eea7b3803fdc3e5825e6d1bb7349a67c9d0450a6d119f13ba87e6ef0864"
NATURAL_LOG = "dc05de3044b9371261df2e211a9948b1dad0d9cc0f9468242d736fd8dc2a6848"


def main(mode,index):
    assert torch.__version__ == "2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    if sha("/stock.zip") != ARCHIVE or sha("/stress.json") != STRESS or sha("/natural.log") != NATURAL_LOG:
        raise ValueError("native archive, originating threshold test, or natural veto changed")
    original = json.loads(Path("/stress.json").read_text())
    assert original["passed"] and not original["held_labels_opened"]
    x,_,chans,order,groups = inputs()
    if sha("/proof.npz") != CANARY:
        raise ValueError("the original paired geometry did not authenticate")
    person = tuple(groups)[index]
    rows = groups[person]
    assert 186 <= len(rows) <= 199
    if mode == "smoke":
        assert person == "P01-1"
    else:
        receipt = Path("tape-smoke.npz")
        assert sha(receipt) == receipt.with_suffix(".sha256").read_text().strip()
        with np.load(receipt,allow_pickle=False) as z:
            p = json.loads(str(z["metadata"]))
        assert p["mode"] == "smoke" and p["source"] == sha("source.sha256") and p["canary"] == CANARY
        assert p["conditioning_forecast_passed"] and not p["held_labels_opened"]
    with TemporaryDirectory() as root:
        with ZipFile("/stock.zip") as zip_file:
            zip_file.extractall(root)
        template = replay.load(Path(root),chans)
        assert type(template).__name__ == "Ensemble6" and template.routed == 1
        assert template.r15 and not template.v6r and template.ft.cap == 250
        gates = (template.route.identity,template.route.contrast)
        config = [g.tau for g in gates]
        with torch.jit.optimized_execution(False):
            natural,n,events = stream(template,x,rows)
            forced,f,forced_events = stream(template,x,rows,suppressed=True)
            for arm,tape in ((natural,events),(forced,forced_events),(natural,events)):
                prefix,_ = private(template,tape)
                np.testing.assert_array_equal(prefix,arm["stages"][:17])
        assert [g.tau for g in gates] == config and not template.cp.t and not template.ft.steps
        assert n["input_threshold"] == f["input_threshold"] == n["active_threshold"] == config
        assert f["active_threshold"] == [1e6,1e6]
    a,b = n["work"],f["work"]
    ages = list(range(20,len(rows),10))
    if (a["recognized"] != 10 or n["warm_rows"] or a["ft_steps"]
            or b["recognized"] != -1 or f["warm_rows"] != len(rows)-20 or b["ft_steps"] != 20*len(ages)
            or b["ft_refits"] != len(ages) or b["ft_relabels"] != len(ages) or b["cp_refits"] < 5 or b["cp_fused"] <= 0
            or f["actual_refit_ages"] != ages or min(n["movement"],f["movement"]) <= .001):
        raise ValueError("unexpected genuine natural or suppressed-recognition workload")
    for key in ("rows","age","weights","rest"):
        np.testing.assert_array_equal(natural[key],forced[key])
    assert n["x_sha"] == f["x_sha"]
    np.testing.assert_array_equal(natural["recognized"],np.arange(len(rows))>=10)
    assert not forced["recognized"].any()
    for key in ("stock_class","upstream_class"):
        np.testing.assert_array_equal(natural[key][:10],forced[key][:10])
    drift = float(np.abs(natural["frozen"]-forced["frozen"]).max())
    if drift > 5e-5:
        raise ValueError("threshold-only intervention changed frozen source member evaluation")
    protected = {k:float(np.abs(v-forced[k]).max()) for k,v in natural.items() if k.startswith("protected_")}
    if len(protected) != 7 or not all(np.isfinite(d) and d<=5e-5 for d in protected.values()):
        raise ValueError("recognition intervention changed protected upstream components")
    np.testing.assert_array_equal(forced["active"],np.arange(len(rows))>=20)
    for arm in (natural,forced):
        for key in protected:
            del arm[key]
    sigma = f["gradient"]["warm_tangent"]
    pass_rank = sigma["rank2"] >= .90*sigma["rows"] and sigma["median_sigma2"] >= 1e-4
    def changes(key):
        return np.flatnonzero(natural[key]!=forced[key]).tolist()
    report = dict(source=sha("source.sha256"),data=DATA,delivery=DELIVERY,archive=ARCHIVE,full0=stamp(order),
                  rows=stamp(rows),scope="paired natural and threshold-suppressed public geometry",result="geometry_only",
                  mode=mode,index=index,session=person,canary=CANARY,protected_max=protected,
                  natural=n,forced=f,stress=STRESS,natural_veto=NATURAL_LOG,held_labels_opened=False,external_opened=False,
                  work_identity_input=True,interleaved_prefix=True,conditioning_forecast_passed=pass_rank,
                  frozen_independent_max=drift,upstream_changed_ages=changes("upstream_class"),
                  terminal_changed_ages=changes("stock_class"),
                  max_mix_delta=float(np.abs(natural["stock_mix"]-forced["stock_mix"]).max()),
                  replaceable_nominal_mass=float((1-natural["rest"])*natural["weights"][1:].sum()))
    fields = {f"{name}|{k}":v for name,arm in (("natural",natural),("forced",forced)) for k,v in arm.items()}
    if mode == "smoke":
        if not pass_rank:
            raise ValueError("P01 warm tangent did not reproduce qualification")
        with np.load("/proof.npz",allow_pickle=False) as z:
            for a,values in (("natural",natural),("forced",forced)):
                np.testing.assert_array_equal(z[f"{a}|stock_class"],values["stock_class"])
                np.testing.assert_array_equal(z[f"{a}|rows"],rows)
            before = json.loads(str(z["metadata"]))
        for a in ("natural","forced"):
            assert report[a]["work"] == before[a]["work"]
    out = Path("tape-smoke.npz" if mode == "smoke" else f"tape-{index}.npz")
    digest = write(out,fields,report)
    print("SEALED F-C31c source",person,mode,digest,"rank gate",pass_rank,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("smoke","capture"))
    p.add_argument("index",type=int,choices=range(14))
    a = p.parse_args()
    main(a.mode,a.index)
