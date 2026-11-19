"""Authenticate the all-PROTEUS F16e hard recipes before the new fit."""
import json
from pathlib import Path

import numpy as np

from anchors import DATA_SHA,ORIGIN_SHA,RUNTIME,row_sha,schedule,sha


HERE=Path(__file__).resolve().parent
F32_ACTOR="f8d074e6867ebfda82383c164c257d64f154b251b0ccdbcb5153b297e7917723"
F16E_ACTOR="b4dd2f3eb7ae50c417334fed30d4a5f06553e28631892245e0e456983601441a"
F16E_LOCK="bc383cbc561223a803ef597300de5cde5869034dc35e464f0852462a2593ca94"
F16E_HARD={
    (0,True):("96b24c78cdcfc86e0a6940661fd283b4b79bd6bd8d0f992b5f64d4500203b5e2",
              "af071e53f3e82647c02c59a49b2cf38448265057ca0f09e4ce6907c245173935"),
    (0,False):("bcedeb0fb01b1735a834f9c8a7270d04f97fe085abce31a5c0bb05d21aa275f5",
               "5c1a9e1e9733c03bad37f89db66781cc894403a5346d4f1b4afb9af74286813c"),
    (1,False):("82b9ae356bc5e09131f3b5db3ced8674f1691e1839031c78e020844df233781d",
               "5879359f057a2a095b99500348abc834b16009b81d36af3d762ced19f4a936e3"),
}


def scope(path):
    from external_source import all_source

    x,rows,streams,people=all_source(path,waveforms=True)
    with np.load(path,allow_pickle=False) as archive:
        channels=archive["ch_names"]
    if len(channels)!=41 or len(np.unique(streams))!=14:
        raise ValueError("all-source channel order or session roster differs")
    return x,rows,streams,people,channels


def authenticated(root, directory, seed, smoke, rows, stream, people):
    if sha(root/"actor.sha256")!=F16E_ACTOR:
        raise ValueError("F16e all-source actor differs")
    tag=f"{'smoke_' if smoke else ''}{seed}_all_source"
    path=directory/f"hard_{tag}.pt"
    receipt=path.with_suffix(".json")
    expected_checkpoint,expected_receipt=F16E_HARD[seed,smoke]
    if sha(path)!=expected_checkpoint or sha(receipt)!=expected_receipt:
        raise ValueError("pinned F16e hard artifact differs")
    base=json.loads(receipt.read_text())
    epochs=4 if smoke else 40
    batches, trace=schedule(rows,stream,seed,epochs)
    expected=dict(held="all_source",kind="hard",seed=seed,epochs=epochs,steps=43*epochs,actor=F16E_ACTOR,
                  dataset=DATA_SHA,origin=ORIGIN_SHA,runtime=RUNTIME,source_rows=row_sha(rows),
                  source_people=people,batches=trace,alpha=0.,parameters=347903,q_gradients_nonzero=0,
                  held_labels_opened=False,source_only_labels=True,checkpoint=expected_checkpoint)
    if (any(base[k]!=v for k,v in expected.items()) or sum(map(len,batches))!=base["steps"] or
            not np.isfinite(base["scale"]) or base["scale"]<=0 or base["state"]==base["initial"]):
        raise ValueError("F16e all-PROTEUS hard recipe or row scope differs")
    return base, batches
