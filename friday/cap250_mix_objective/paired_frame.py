"""Pair natural and threshold-suppressed source sessions, without scoring."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

import replay
from source_frame import ARCHIVE, DATA, DELIVERY, inputs, private, sha, stamp, stream, write


STRESS = "95a1fc3d4bf3f0ddd2491f90b6181f945f3446b093c9814f65cc387deaacfe4a"
NATURAL_LOG = "dc05de3044b9371261df2e211a9948b1dad0d9cc0f9468242d736fd8dc2a6848"


def main():
    assert torch.__version__ == "2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    if sha("/stock.zip") != ARCHIVE or sha("/stress.json") != STRESS or sha("/natural.log") != NATURAL_LOG:
        raise ValueError("native archive, originating threshold test, or natural veto changed")
    original = json.loads(Path("/stress.json").read_text())
    assert original["passed"] and not original["held_labels_opened"]
    x,_,chans,order,groups = inputs()
    rows = groups["P01-1"]
    assert len(rows) == 190
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
    if (a["recognized"] != 10 or n["warm_rows"] or a["ft_steps"]
            or b["recognized"] != -1 or f["warm_rows"] != 170 or b["ft_steps"] != 340
            or b["ft_refits"] != 17 or b["ft_relabels"] != 17 or b["cp_refits"] < 5 or b["cp_fused"] <= 0
            or f["actual_refit_ages"] != list(range(20,190,10)) or min(n["movement"],f["movement"]) <= .001):
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
    np.testing.assert_array_equal(forced["active"],np.arange(len(rows))>=20)
    sigma = f["gradient"]["warm_tangent"]
    pass_rank = sigma["rank2"] >= .90*sigma["rows"] and sigma["median_sigma2"] >= 1e-4
    def changes(key):
        return np.flatnonzero(natural[key]!=forced[key]).tolist()
    report = dict(source=sha("source.sha256"),data=DATA,delivery=DELIVERY,archive=ARCHIVE,full0=stamp(order),
                  rows=stamp(rows),scope="paired natural and threshold-suppressed public geometry",result="geometry_only",
                  natural=n,forced=f,stress=STRESS,natural_veto=NATURAL_LOG,held_labels_opened=False,external_opened=False,
                  work_identity_input=True,interleaved_prefix=True,conditioning_forecast_passed=pass_rank,
                  frozen_independent_max=drift,upstream_changed_ages=changes("upstream_class"),
                  terminal_changed_ages=changes("stock_class"),
                  max_mix_delta=float(np.abs(natural["stock_mix"]-forced["stock_mix"]).max()),
                  replaceable_nominal_mass=float((1-natural["rest"])*natural["weights"][1:].sum()))
    fields = {f"{name}|{k}":v for name,arm in (("natural",natural),("forced",forced)) for k,v in arm.items()}
    digest = write(Path("paired-geometry.npz"),fields,report)
    print("SEALED F-C31b geometry",digest,"rank gate",pass_rank,flush=True)


if __name__ == "__main__":
    main()
