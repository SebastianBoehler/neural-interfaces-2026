"""Read only caller-visible EEG and fixed row identities for the two exact passes."""

from pathlib import Path
import json

import numpy as np
import torch

import replay
from settings import SEEDS, schedule


def layout(mode,index):
    with np.load("/external.npz",allow_pickle=False) as z:
        eeg=torch.from_numpy(z["X"])
        people,visits,channels=z["subject"],z["session"],z["ch_names"].tolist()
    assert eeg.shape==(2130,41,480) and eeg.dtype==torch.float32
    who=sorted(set(people))[index]
    plans=schedule(people,visits)
    path=Path("/rtx_pair" if index in (0,5) else "/rtx_other")/f"rtx_cap_{index}.json"
    old=json.loads(path.read_text())
    assert (old["arm"],old["person"])==("cap",index)
    anchor=old["sessions"][f"{who}|1|0"]
    assert plans[0][who+"|1"].tolist()==anchor["rows"]
    anchors=((eeg,plans[0][who+"|1"]),anchor,path)
    choices={}
    if mode=="smoke":
        assert who=="Scherer2015/G"
        pub,days,names=replay.read_source()
        assert names==channels
        witness=json.loads(Path("/known_rtx.json").read_text())["shared"]["recognized"]
        rows=dict(days)["P10-3"][:31]
        assert rows.tolist()==witness["rows"] and witness["activity"]["recognized"]==10
        choices[f"{who}|1|0"]=(eeg,plans[0][who+"|1"])
        choices["P10-3"]=(pub,rows)
    else:
        for seed in SEEDS:
            for day in (1,2):
                key=f"{who}|{day}"
                choices[f"{key}|{seed}"]=(eeg,plans[seed][key])
    return choices,anchors,channels,who
