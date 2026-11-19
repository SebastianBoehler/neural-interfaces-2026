"""Check the exact F33 source recipe on Galvani without held inference."""
import json
from pathlib import Path

import braindecode
import numpy as np
import torch

from anchors import DATA_SHA, row_sha, sha, split
from covariance import BASE, KINDS, MAX_RATIO, MAX_RESERVED, trainable
from encoder312 import fingerprint
import model322
from phase_source import authenticated
from source_checks import diagnostics, fit, install
from source_labels import read_training_labels
from source_targets import normalizer


BASE_PATH=Path("/mnt/lustre/work/gehler/gfh527")
ROOT=BASE_PATH/"ni26_s323_normsplit_20261117"
F16=BASE_PATH/"ni26_friday_s1c_distill_20261118"


def model(seed):
    model322.SEED=seed
    out=model322.Factor("session",False,"cuda")
    normalizer(out)
    if trainable(out)!=BASE:
        raise ValueError("invalid same-device hard architecture")
    return out


def median_time(actor,sample):
    import time

    with torch.no_grad():
        for _ in range(4):
            actor(sample)
        timings=[]
        for _ in range(5):
            torch.cuda.synchronize()
            start=time.perf_counter()
            actor(sample)
            torch.cuda.synchronize()
            timings.append(1000*(time.perf_counter()-start))
    return float(np.median(timings))


def run():
    device=torch.cuda.get_device_name()
    if ("A100" not in device or str(torch.__version__)!="2.14.1+cu126" or braindecode.__version__!="1.8.1" or
            Path(model322.__file__).resolve()!=ROOT/"model322.py"):
        raise ValueError("unverified same-recipe A100 runtime")
    held,seed="P01",0
    torch.set_num_threads(2)
    path=ROOT/"data/proteus_windows.npz"
    if sha(path)!=DATA_SHA:
        raise ValueError("wrong original raw tensor")
    with np.load(path,allow_pickle=False) as z:
        eeg,subject,visit=(z[k] for k in ("X","subject","session"))
    if eeg.shape!=(2655,41,480) or eeg.dtype!=np.float32 or not np.isfinite(eeg).all():
        raise ValueError("wrong untouched EEG shape")
    source,streams,people=split(subject,visit,held)
    baseline,batches=authenticated(F16,F16/"checkpoints",held,seed,True,source,streams,people)
    labels=read_training_labels(path,source,len(eeg))
    x,y,scale,mu,sd,weight,observed=model322.setup(dict(X=eeg[source],y=labels,stream=streams),
                                                   np.arange(len(source)),"cuda",epochs=baseline["epochs"])
    flat=[rows for ep in batches for rows in ep]
    if (not np.isclose(scale,baseline["scale"],atol=0,rtol=2e-6) or
            any(not np.array_equal(p,q) for p,q in zip(flat,(r for ep in observed for r in ep),strict=True))):
        raise ValueError("alternate source batching or normalized scale differs")
    source_args=x,y,mu,sd,weight,batches
    native=model(seed)
    initial=fingerprint(native)
    reference_stats=model322.train(native,*source_args)
    reference=fingerprint(native)
    paired=dict(baseline,initial=initial,state=reference)
    if reference_stats["steps"]!=baseline["steps"] or reference==initial:
        raise ValueError("missing same-device native reference")
    records={}
    initial_basis=None
    hard_ms=None
    sample=x[flat[0][:32]]
    for kind in KINDS:
        actor=model(seed)
        if fingerprint(actor)!=initial:
            raise ValueError("same-device match lost native initialization")
        layer,proof=install(actor,kind,seed,sample)
        stats=fit(actor,layer,source_args,paired)
        if layer is not None:
            if initial_basis is not None and proof["initial_basis"]!=initial_basis:
                raise ValueError("aligned and permuted A100 bases differ")
            initial_basis=proof["initial_basis"]
            proof.update(diagnostics(actor,layer,sample,kind))
        ms=median_time(actor,sample)
        hard_ms=ms if kind=="hard" else hard_ms
        if not 0<ms<=MAX_RATIO*hard_ms:
            raise ValueError("source paired warm timing exceeds preregistered ratio")
        records[kind]=dict(proof,**stats,warm_ms=ms,parameters=trainable(actor))
    if len({r["state"] for r in records.values()})!=3 or torch.cuda.max_memory_reserved()>MAX_RESERVED:
        raise ValueError("A100 paired learned state or memory budget failed")
    receipt=dict(scope="source only",device=device,torch=str(torch.__version__),cuda=torch.version.cuda,
                 braindecode=braindecode.__version__,cudnn=torch.backends.cudnn.version(),
                 cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                 source=row_sha(source),people=people,steps=baseline["steps"],batches=baseline["batches"],scale=scale,
                 h100_initial=baseline["initial"],h100_state=baseline["state"],a100_initial=initial,a100_state=reference,
                 held_labels_opened=False,actor=sha(Path("canary.sha256")),results=records)
    target=Path("source-receipt.json")
    with target.open("x") as handle:
        handle.write(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(f"PASS F33 A100 source canary {sha(target)}",flush=True)


if __name__=="__main__":
    run()
