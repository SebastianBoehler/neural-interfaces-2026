"""Bind F33a to original source recipes and the completed A100 canary."""
import json
from pathlib import Path

import numpy as np

from anchors import DATA_SHA,ORIGIN_SHA,RUNTIME,row_sha,schedule,sha
from phase_source import F16_ACTOR,F16_PINS


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/"ni26_s323_normsplit_20261117"
F16=HERE.parent/"ni26_friday_s1c_distill_20261118"
F17=HERE.parent/"ni26_friday_s1c_phase_20261118"
CANARY=HERE.parent/"ni26_friday_s1c_covariance_a100_20261119"/"source-receipt.json"
CANARY_SHA="16ce19d2812306824f4bf565a749ab95eb5e88493056ca26cc34356f5a66c556"
F33="984e6a7c7776fefca6fd70924ddc135b92dd53ab3fdffd206fc435ccaa5df70f"
GPU="NVIDIA A100-PCIE-40GB"
TORCH="2.14.1+cu126"
BD="1.8.1"
CPU_CUDA=("cudnn","cudnn_tf32","matmul_tf32")


def tag(held,seed,smoke):
    return f"{'smoke_' if smoke else ''}{seed}_{held}"


def historical(held,seed,smoke,rows,streams,people):
    pins=F17/"f16_pins.json"
    if sha(F16/"actor.sha256")!=F16_ACTOR or sha(pins)!=F16_PINS:
        raise ValueError("original F16 recipe ancestry differs")
    name="hard_"+tag(held,seed,smoke)
    pin=json.loads(pins.read_text())[name]
    receipt=F16/"checkpoints"/f"{name}.json"
    if sha(receipt)!=pin["receipt"]:
        raise ValueError("historical hard recipe metadata differs")
    meta=json.loads(receipt.read_text())
    batches,trace=schedule(rows,streams,seed,4 if smoke else 40)
    fixed=dict(kind="hard",held=held,seed=seed,epochs=len(batches),dataset=DATA_SHA,origin=ORIGIN_SHA,runtime=RUNTIME,
               source_rows=row_sha(rows),source_people=people,actor=F16_ACTOR,batches=trace,state=pin["state"],
               checkpoint=pin["checkpoint"],q_gradients_nonzero=0,held_labels_opened=False,parameters=347903,
               steps=sum(map(len,batches)),braindecode=BD,torch=TORCH)
    if any(meta[k]!=v for k,v in fixed.items()) or not np.isfinite(meta["scale"]) or meta["scale"]<=0:
        raise ValueError("authenticated original source fit recipe differs")
    return meta,batches,pin["receipt"]


def canary():
    if sha(CANARY)!=CANARY_SHA:
        raise ValueError("original A100 source canary differs")
    m=json.loads(CANARY.read_text())
    if (m["device"]!=GPU or m["torch"]!=TORCH or m["braindecode"]!=BD or m["held_labels_opened"]
            or m["a100_initial"]!=m["h100_initial"] or m["a100_state"]==m["h100_state"]):
        raise ValueError("previous A100 source identity differs")
    return m
