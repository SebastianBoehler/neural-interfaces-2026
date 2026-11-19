"""Run each immutable RTX archive diagnosis independently of the other fixture."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from same_live import direct_trace, native_mirrors
from smoke import ZIP_HASH, load_solver
from smoke_gate import REFERENCE_SHA, digest, frames, fresh_model, load_paths, pinned
from stock_drift import aa, activity, effects


ARCHIVE = "cf4a8deee46a25e59c2bb664bb0761254a6368467ac3eee5dff6c0bb572e447f"
P10_LABELS = "e687cd98794957f3590884ef57f1af7203563edd07b01b0cad4d2bd4baa0b569"


def check_live(ref, stock, model, windows):
    labels, live = direct_trace(model,windows)
    own, private, selected = live.stages()
    old, alt, baseline, candidate = native_mirrors(ref,stock,live.events)
    np.testing.assert_array_equal(np.stack([e[0] for e in live.events]),windows.cpu().numpy())
    mask = (np.arange(len(windows))>=20) & ~np.array([e[7] for e in live.events])
    y = np.where(mask[:,None],alt[:,2],old[:,2])
    recognized = int(live.recognition_age)
    n = len(windows) if recognized < 0 else recognized
    np.testing.assert_array_equal(own,old)
    np.testing.assert_array_equal(private,alt[:n])
    np.testing.assert_array_equal(selected,y)
    np.testing.assert_array_equal(labels,y.argmax(1))
    p = live.summary()
    if (p["alternate_rows"],p["shared_calls"],p["verified_covariances"],p["eligible"],
        p["alternate_refits"],p["alternate_fused"]) != (
            n,n,n,int(mask.sum()),sum(i<n for i in candidate.fit),candidate.cp.fused if recognized<0 else 0):
        raise ValueError("shipped same-input comparison missed actual downstream work")
    if (baseline.cp.refits,baseline.cp.fused) != (model.stock.cp.refits,model.stock.cp.fused):
        raise ValueError("original stock personalizer did different work")
    return labels, dict(stock=own,alternate=private,selected=selected), p


def same(ref, stock, template, windows, peer, expected, work):
    trial = fresh_model(template)
    labels, vectors, proof = check_live(ref,stock,trial,windows)
    base = vectors["stock"][:,2].argmax(1)
    np.testing.assert_array_equal(base,expected)
    if activity(trial.stock)!=work:
        raise ValueError("deployed original stock activity differs from the pinned archived scope")
    resets = {}
    for tag,model in (("active_copy",fresh_model(trial)),("reused_object",trial)):
        model.reset_state()
        model.predict(peer)
        model.reset_state()
        if model.selector is not None:
            raise ValueError("the shipped selector survived native session reset")
        first, later, reset = check_live(ref,stock,model,windows[:1])
        if model.stock.cp.t!=1 or model.stock.cf.t!=1 or reset["rows"]!=1 or reset["recognized"]!=-1:
            raise ValueError("actual shipped worker did not reset all owning clocks")
        compare = effects({k:v[:1] for k,v in vectors.items()},later)
        resets[tag] = dict(same_live_exact=True,decision_exact=bool(np.array_equal(first,labels[:1])),
                           independent_vector_exact=all(x["max"]==0 for x in compare.values()), stages=compare)
    return dict(rows=len(windows),same_live_exact=True,stock_work=work,shipped=proof,
                changed_vs_stock=int(np.count_nonzero(base!=labels)),interleaved_resets=resets)


def inputs(args):
    fixtures = dict(frames(args.public,args.external,args.known,args.unknown,external_rows=120))
    p10 = json.loads(args.known.read_text())["shared"]["recognized"]
    g = json.loads(args.unknown.read_text())["sessions"]["Scherer2015/G|1|0"]
    size = {"recognized":31,"unrecognized":120}
    if any(len(x)!=size[k] for k,x in fixtures.items()):
        raise ValueError("the original delivery scope changed")
    normal = json.loads(pinned(args.predicted, P10_LABELS).read_text())["streams"]["first"]["P10-3"]
    if normal["activity"]["recognized"] != p10["activity"]["recognized"]:
        raise ValueError("stock recognition differs across pinned prediction receipts")
    expected=normal["labels"][:31] if args.fixture=="recognized" else g["labels"]
    work=p10["activity"] if args.fixture=="recognized" else g["activity"]
    name = args.fixture
    other = "unrecognized" if name=="recognized" else "recognized"
    return fixtures[name].to("cuda:0"), fixtures[other][:1].to("cuda:0"), fixtures["unrecognized"].to("cuda:0"), expected, work


def main(args):
    if (torch.__version__ != "2.14.1+cu130" or "RTX 2080 Ti" not in torch.cuda.get_device_name()
            or "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text()):
        raise ValueError("original clean RTX worker required")
    torch.set_num_threads(1)
    pinned(args.stock,ZIP_HASH)
    pinned(args.trial,ARCHIVE)
    ref={"np":np,"torch":torch}
    exec(compile(pinned(args.reference,REFERENCE_SHA).read_text(),"<signed-original-live-source>","exec"),ref)
    windows,peer,warm,expected,work = inputs(args)
    with TemporaryDirectory() as scratch:
        if args.mode=="aa":
            with ZipFile(args.stock) as z:
                z.extractall(scratch)
            stock = load_solver(Path(scratch),"cuda:0")
            trial = None
        else:
            stock,trial = load_paths(args.stock,args.trial,scratch,"cuda:0")
        initial=fresh_model(stock)
        for row in warm:
            initial.predict(row[None])
        del initial
        checked=aa(ref,stock,windows,peer,expected,work) if args.mode=="aa" else same(ref,stock,trial,windows,peer,expected,work)
    here=Path(f"{args.mode}_{args.fixture}.json")
    with here.open("x") as f:
        json.dump(dict(source=digest(Path("source.sha256")),prior="b0bc9a635389c4102107dabb9ceb7f13fc4f3a3d9f5ec2b3477160c0d267cc37",
                       archive=ARCHIVE,stock=ZIP_HASH,mode=args.mode,fixture=args.fixture,held_labels_opened=False,
                       device=torch.cuda.get_device_name(),observations=checked),f,sort_keys=True,allow_nan=False)
        f.write("\n")
    here.chmod(0o444)
    print("COMPLETE F-C21gpackD",args.mode,args.fixture,digest(here),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("aa","same"))
    p.add_argument("fixture",choices=("recognized","unrecognized"))
    for name in ("stock","trial","public","external","known","unknown","reference","predicted"):
        p.add_argument(name,type=Path)
    main(p.parse_args())
