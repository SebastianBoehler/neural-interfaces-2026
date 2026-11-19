"""Read and schedule the original authenticated F6 PhysioNet exports."""
from pathlib import Path

import numpy as np
import torch

from anchors import row_sha, sha


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "detached", "joint")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{k}_{m}" for k in KINDS for m in MODES)
CLASSES = ("feet", "hands", "left_hand", "rest", "right_hand")
PEOPLE = [f"PhysionetMI/{i}" for i in range(1, 31)]
SHARDS = {
    "windows_1-10.npz":"9fcdb9d8535dad6e38bcbee39095209a4d8490790bac491b0bd11c50e2f621c4",
    "windows_11-20.npz":"2342ac474be4edb168887b9a0396f737c690795d76b77e5b02c89219b3a3acda",
    "windows_21-30.npz":"7cddfb92000a2efadb513bb43fe386c02c7397bc84ab20e8fa51367d4d0bb5ae",
}
F6_INPUTS = "b49b01b84ed61ad10865381a68c49d8e65b23ce99ed79527fc002071ad66a403"
WEIGHT, BATCH_SIZE, PREFIX, EVENT, TOL = .25,40,240,360,2e-5


def metadata(root,channels):
    people, labels, runs, onsets = [], [], [], []
    for name in SHARDS:
        with np.load(root/name,allow_pickle=False) as z:
            if (tuple(z["classes"])!=CLASSES or not np.array_equal(z["ch_names"],channels) or z["interpolated"].size or
                    set(z["task"])!={"PhysionetMI"} or set(z["session"])!={"0"} or set(z["split"])!={"external"}):
                raise ValueError(f"unexpected PhysioNet external schema: {name}")
            people.append(z["subject"])
            labels.append(z["y"])
            runs.append(z["run"])
            onsets.append(z["onset"])
    person,y,run,onset=(np.concatenate(v) for v in (people,labels,runs,onsets))
    if (len(y)!=5220 or set(person)!=set(PEOPLE) or set(run)!=set("012345") or
            not np.isfinite(onset).all() or set(y)!=set(range(5))):
        raise ValueError("external waveform cohort or motor-cue label scope differs")
    for p in PEOPLE:
        for r in "012345":
            chosen=(person==p)&(run==r)
            expected={2,3,4} if int(r)<3 else {0,1,3}
            if (chosen.sum()!=29 or set(y[chosen])!=expected or len(np.unique(onset[chosen]))!=29):
                raise ValueError(f"run-aware imagined motor/rest mapping differs for {p}/{r}")
    return person,y


def waveforms(folder):
    files=[]
    f6=HERE.parent/"friday_wave_motor_20261118/pretrain/physionet.sha256"
    if sha(f6)!=F6_INPUTS or {name:digest for digest,name in (v.split() for v in f6.read_text().splitlines())}!=SHARDS:
        raise ValueError("F6 external-data manifest differs")
    for name,digest in SHARDS.items():
        if sha(folder/name)!=digest:
            raise ValueError(f"full PhysioNet shard bytes differ: {name}")
        with np.load(folder/name,allow_pickle=False) as z:
            files.append(z["X"])
    x=np.concatenate(files)
    if x.shape!=(5220,41,480) or x.dtype!=np.float32 or not np.isfinite(x).all():
        raise ValueError("external waveform shape or finite values differ")
    tensor=torch.from_numpy(x).cuda()
    scale=float(tensor[...,:EVENT].square().mean().sqrt())
    if not np.isfinite(scale) or not scale>0:
        raise ValueError("invalid private PhysioNet scale")
    return tensor/scale,scale


def tape(person,y,held,seed,steps):
    rng=np.random.default_rng(32032+100*int(held[1:])+seed)
    grid=[[np.flatnonzero((person==p)&(y==j)) for p in PEOPLE] for j in range(5)]
    if any(len(v)==0 for choices in grid for v in choices):
        raise ValueError("source cohort lacks the declared motor/rest exposures")
    chosen=np.empty((steps,40),dtype=np.int64)
    for i in range(steps):
        pieces=[]
        for group in grid:
            names=rng.choice(len(PEOPLE),size=8,replace=False)
            pieces.extend(rng.choice(group[j]) for j in names)
        chosen[i]=pieces
    if not np.array_equal(y[chosen],np.broadcast_to(np.repeat(np.arange(5),8),chosen.shape)):
        raise ValueError("paired external batch labels differ")
    return chosen,dict(external_tape=row_sha(chosen),external_rows=len(y),external_subjects=PEOPLE,
                        external_shards=SHARDS,external_batches=steps,external_exposure=steps*BATCH_SIZE,
                        external_classes=list(CLASSES),aux_temporal_samples=PREFIX,event_samples=EVENT)
