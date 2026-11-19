"""Tap the original public Solver without reading the task-label field."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

import replay
from adapter import Adapter
from benchmark_utils.streaming import fresh_model
from mirror import Outer
from mixture import IDS, TOL, candidate, gradient, stock_mix


DATA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
DELIVERY = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
ARCHIVE = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
FIXTURES = ("P01-1","P10-3")
KEY = "full|0|all_loso"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def stamp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def inputs():
    if sha("/input.npz") != DATA or sha("/orders.npz") != DELIVERY:
        raise ValueError("original PROTEUS windows or delivery archive changed")
    with np.load("/input.npz",allow_pickle=False) as a:
        x = torch.from_numpy(a["X"])
        people = np.array([s.rsplit("/",1)[-1] for s in a["subject"]])
        names = np.char.add(np.char.add(people,"-"),a["session"].astype(str))
        chans = list(a["ch_names"])
    with np.load("/orders.npz",allow_pickle=False) as a:
        order = a[KEY]
    np.testing.assert_array_equal(np.sort(order),np.arange(2655))
    groups = {name:order[names[order]==name] for name in sorted(set(names))}
    assert x.shape == (2655,41,480) and x.dtype == torch.float32 and len(chans) == 41
    assert len(set(people)) == 10 and len(groups) == 14 and all(np.any(np.diff(r)<0) for r in groups.values())
    return x,names,chans,order,groups


class SourceTap:
    def __init__(self,model):
        self.model = model
        self.weights = model.w[1+np.asarray(IDS)]
        self.rest = float(model.w_rest)
        self.native,self.route,self.terminal = model.mix6,model.routeB.stepx,model.cp.step
        self.pending = None
        self.events,self.operands,self.stages = [],[],[]
        if (tuple(model.ft.slots["fbraw"]) != IDS or model.ft.member["fbraw"] != IDS[0]
                or not 0 < self.rest < 1):
            raise ValueError("original raw family, seed or Rest mass differs")
        np.testing.assert_allclose(self.weights,1/90,atol=1e-16,rtol=0)
        model.mix6,model.routeB.stepx,model.cp.step = self.mix,self.consume,self.final

    def mix(self,mix,members,warm):
        if self.pending is not None or not members:
            raise ValueError("protected native mixer call order or inventory differs")
        base = self.native(mix,members,warm)
        np.testing.assert_array_equal(base,stock_mix(self.model,mix,members,warm))
        np.testing.assert_array_equal(base,self.native(mix,members.copy(),warm))
        f = np.stack([members[k] for k in IDS])
        adapted = warm.get("fbraw",f[0])
        active = "fbraw" in warm
        probes = []
        for q in (f[1:].mean(0),f[1],f[-1]):
            delta = (self.weights[1:,None]*(q-f[1:])).sum(0)
            new = dict(members)
            new.update({i:q for i in IDS[1:]})
            actual = self.native(mix+delta,new,warm)
            independently = candidate(base,f,adapted,active,self.weights,self.rest,q)
            np.testing.assert_allclose(actual,independently,atol=TOL,rtol=0)
            probes.append(actual)
        identity,shift48,shift52 = probes
        np.testing.assert_allclose(identity,base,atol=TOL,rtol=0)
        self.pending = base
        self.operands.append((base.copy(),f,adapted.copy(),active,shift48,shift52))
        return base

    def consume(self,x,u,m,core,extra,gc,gi):
        if m is not self.pending:
            raise ValueError("actual RouteB bypassed the bound native mixture row")
        self.events.append(deepcopy((x,u,m,m,core,extra,gc,gi,self.model.recB)))
        self.pending = None
        return self.route(x,u,m,core,extra,gc,gi)

    def final(self,x,raw,count,recognized):
        final = self.terminal(x,raw,count,recognized)
        self.stages.append(np.stack((raw,count,final)))
        return final


def private(template,events):
    tail = Outer(template)
    for event in events:
        tail.step(event,"stock")
    p = tail.result()
    return np.stack([p[k] for k in ("raw","count","terminal")],axis=1),tail


def stream(template,x,rows,suppressed=False):
    model = fresh_model(template)
    if model.cp.t or model.cf.t or model.v5hist:
        raise ValueError("source session inherited adaptive state")
    gates = (model.route.identity,model.route.contrast)
    original_threshold = [g.tau for g in gates]
    assert all(0 <= t < 1e6 for t in original_threshold)
    if suppressed:
        for gate in gates:
            gate.tau = 1e6
    monitor = Adapter(model,"stock",())
    tap = SourceTap(model)
    labels, upstream, recognition = [], [], []
    for r in rows:
        p = model.predict(x[int(r):int(r)+1].to("cuda:0"))
        assert p.shape == (1,) and p.device.type == "cpu" and p.dtype == torch.int64
        labels.append(int(p[0]))
        upstream.append(int(model.trace["out_v5"].argmax()))
        recognition.append(bool(model.recB))
    work = replay.activity(model)
    assert work["windows"] == len(rows) == len(tap.events) == len(tap.operands) == len(tap.stages)
    assert work["errors"] == work["ft_over"] == 0 and tap.pending is None
    assert not any(s.dead for s in (model,model.cp,model.cf))
    np.testing.assert_array_equal(np.stack([r[0] for r in tap.events]),x[rows].numpy())
    stages = np.stack(tap.stages)
    mirror,tail = private(template,tap.events)
    np.testing.assert_array_equal(stages,mirror)
    np.testing.assert_array_equal(stages[:,2].argmax(1),labels)
    assert tail.cp.refits == work["cp_refits"] and tail.cp.fused == work["cp_fused"]
    prefix, _ = private(template,tap.events[:17])
    np.testing.assert_array_equal(stages[:17],prefix)
    a = dict(zip(("stock_mix","frozen","warm","active","probe48","probe52"),
                 map(np.asarray,zip(*tap.operands,strict=True)),strict=True))
    assert a["active"].dtype == bool and a["active"].sum() == model.log6["ft"]
    assert np.isfinite(a["frozen"]).all() and (a["frozen"]>=0).all()
    np.testing.assert_allclose(a["frozen"].sum(2),1.,atol=1e-12,rtol=0)
    proof = gradient(a,tap.weights,tap.rest)
    move = max(float(np.abs(a[k]-a["stock_mix"]).max()) for k in ("probe48","probe52"))
    info = dict(work=work, gradient=proof, movement=move, warm_rows=int(a["active"].sum()),
                all_stages_zero_exact=True,prefix_exact=True,x_sha=stamp(x[rows].numpy()),
                input_threshold=original_threshold,active_threshold=[g.tau for g in gates],
                actual_refit_ages=[r["age"] for r in monitor.refits],
                refit_receipt=hashlib.sha256(json.dumps(monitor.refits,sort_keys=True).encode()).hexdigest())
    return dict(a, rows=rows, age=np.arange(len(rows)), weights=tap.weights, rest=np.array(tap.rest),
                stages=stages, stock_class=np.array(labels), upstream_class=np.array(upstream),
                recognized=np.array(recognition)), info, tap.events[:17]


def write(path,fields,metadata):
    with path.open("xb") as f:
        np.savez_compressed(f,metadata=json.dumps(metadata,sort_keys=True,allow_nan=False),**fields)
    path.chmod(0o444)
    with path.with_suffix(".sha256").open("x") as f:
        f.write(sha(path)+"\n")
    path.with_suffix(".sha256").chmod(0o444)
    return sha(path)


def main(mode,index):
    if "RTX 2080 Ti" not in torch.cuda.get_device_name() or torch.__version__ != "2.14.1+cu130":
        raise ValueError("requires the qualified native cap250 RTX image")
    torch.set_num_threads(1)
    x,_,chans,order,groups = inputs()
    if sha("/stock.zip") != ARCHIVE:
        raise ValueError("original frozen cap250 archive changed")
    source = sha("source.sha256")
    common = dict(source=source,archive=ARCHIVE,data=DATA,delivery=DELIVERY,full0=stamp(order),
                  external_opened=False,held_labels_opened=False,production_native_scope=True)
    with TemporaryDirectory() as root:
        with ZipFile("/stock.zip") as z:
            z.extractall(root)
        template = replay.load(Path(root),chans)
        assert type(template).__name__ == "Ensemble6" and template.routed == 1
        assert template.r15 and not template.v6r and template.ft.cap == 250
        if mode == "smoke":
            reports = {}
            first_prefix = None
            for name in FIXTURES:
                a,info,events = stream(template,x,groups[name])
                if info["movement"] <= .001:
                    raise ValueError("native row-wise mixer probe movement below registered threshold")
                reports[name] = info
                if first_prefix is None:
                    first_prefix = (a["stages"][:17],events)
            pp,_ = private(template,first_prefix[1])
            np.testing.assert_array_equal(pp,first_prefix[0])
            if not reports["P01-1"]["warm_rows"] or reports["P10-3"]["work"]["recognized"] < 0:
                raise ValueError("expected real warmed source and recognized complementary fixtures")
            out = Path("smoke.json")
            with out.open("x") as f:
                json.dump(dict(common,passed=True, interleaved_exact=True,fixtures=reports),f,sort_keys=True,allow_nan=False)
                f.write("\n")
            out.chmod(0o444)
            digest = sha(out)
        else:
            preflight = json.loads(Path("smoke.json").read_text())
            assert preflight["passed"] and preflight["interleaved_exact"]
            assert all(preflight[k] == v for k,v in common.items())
            name = tuple(groups)[index]
            a,info,_ = stream(template,x,groups[name])
            digest = write(Path(f"frame_{index}.npz"),a,dict(common,name=name,index=index,**info))
        assert not template.cp.t and not template.cf.t and not template.v5hist and not template.dead
    print("PASS F-C31a",mode,index,digest,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("smoke","capture"))
    p.add_argument("index",type=int,choices=range(14))
    a = p.parse_args()
    main(a.mode,a.index)
