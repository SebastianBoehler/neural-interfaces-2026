"""Authenticate the exact all-PROTEUS F16e recipe and same-device F25e hard fits."""
import json
from pathlib import Path

import numpy as np

from anchors import DATA_SHA,ORIGIN_SHA,row_sha,sha
from recipe import BD,GPU,TORCH
from source_scope import authenticated,scope


HERE=Path(__file__).resolve().parent
F25E=HERE.parent/"ni26_friday_s1c_modulation_external_a100_20261119"
F16E=HERE.parent/"ni26_friday_s1c_motor_external_cpu_20261119/anchors_f16e"
BUNDLE="7c50ac96002ea129a1bffa6179f1a58174376502147ca2ed4534cb41dc406c50"
SMOKE="04dd2f191a7596562f49a2441d26d53179e4a3d2c6d74703431a4c4b7f7743e7"
F35="b153df05e923c0b40795adaed413d8d024087e49e4b2be51445d81b6180db42e"


def tag(seed,smoke):
    return f"{'smoke_' if smoke else ''}{seed}_all_source"


def source_anchor(seed,smoke,rows,streams,people):
    base,batches=authenticated(F16E,F16E,seed,smoke,rows,streams,people)
    lock=F25E/"artifact.sha256"
    path=F25E/"checkpoints"/f"hard_{tag(seed,smoke)}.pt"
    if sha(lock)!=BUNDLE:
        raise ValueError("independent all-source native A100 bundle differs")
    meta_path=path.with_suffix(".json")
    if smoke:
        receipt=F25E/"smoke_exports/s0.json"
        if seed or sha(receipt)!=SMOKE:
            raise ValueError("independent A100 hard smoke lock differs")
        pins=json.loads(receipt.read_text())["results"]["hard"]
        expected=(pins["checkpoint"],pins["source_receipt"])
    else:
        pins={name:digest for digest,name in (line.split() for line in lock.read_text().splitlines())}
        expected=tuple(pins[str(p.relative_to(F25E))] for p in (path,meta_path))
    if tuple(sha(p) for p in (path,meta_path))!=expected:
        raise ValueError("original same-device hard checkpoint or metadata differs")
    m=json.loads(meta_path.read_text())
    fixed=dict(held="all_source",seed=seed,kind="hard",epochs=base["epochs"],steps=base["steps"],dataset=DATA_SHA,
               origin=ORIGIN_SHA,batches=base["batches"],weights_origin=base["checkpoint"],initial=base["initial"],
               source_rows=row_sha(rows),source_people=people,source_sessions=14,source_count=2655,checkpoint=expected[0],
               h100_state=base["state"],held_labels_opened=False,scherer_exposed=False,device=GPU,torch=TORCH,
               braindecode=BD,parameters=347903,q_gradients_nonzero=0)
    if (any(m[k]!=v for k,v in fixed.items()) or not np.isclose(m["scale"],base["scale"],atol=0,rtol=2e-6) or
            m["state"]!=m["same_device_hard_state"] or m["checkpoint"]!=m["same_device_hard_checkpoint"]):
        raise ValueError("all-source A100 and original F16e recipes differ")
    return base,batches,m
