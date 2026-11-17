"""Fixed source-person/architecture cell; full guarded8-order bank before labels."""
import json
import os
from pathlib import Path
import sys
import time
import torch
import numpy as np
from data354 import source,original,CELLS,ORDERS
from train354 import attach


def main():
    root=Path.cwd();torch.set_num_threads(2);started=time.monotonic();person,arm=CELLS[int(sys.argv[1])]
    smoke=json.loads((root/'smoke.json').read_text())
    if not smoke['passed'] or smoke['source_sha256']!=original.sha(root/'source.sha256'):raise ValueError('same-snapshot smoke required')
    data,fit,held,people,y=source(person);model=original.native(data,40);trace=attach(model,arm);model.fit(fit)
    output,guards=original.inference(model,data,fit,held,y)
    if set(output)!=set(ORDERS):raise ValueError('incomplete8-order output')
    # Cross-session interruption must not alter the next complete session call.
    prior=original.state_sha(model.net);p=model.predict_proba_stream(held)
    model.predict_proba_stream(fit[:7]);q=model.predict_proba_stream(held)
    if not np.array_equal(p,q) or prior!=original.state_sha(model.net):raise ValueError('interleaved-session state guard')
    folder=root/'output'/person/arm;folder.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state=model.net.state_dict(),G=model.G,scale=model.scale,trace=trace),folder/'model.pt')
    with (folder/'predictions.npz').open('xb') as stream:np.savez_compressed(stream,index=held,**output)
    record=dict(person=person,arm=arm,source_people=sorted(set(people[fit])),trace=trace,
        source_rows_sha256=original.digest(fit),query_rows_sha256=original.digest(held),
        source_sha256=original.sha(root/'source.sha256'),model_sha256=original.sha(folder/'model.pt'),
        prediction_sha256=original.sha(folder/'predictions.npz'),guards=guards,
        seconds=time.monotonic()-started,node=os.uname().nodename,query_labels_opened=False,
        scope='INDIVIDUAL-MODEL pilot; exposed P02/P03/P09; source9/seed0; no full-system gain')
    (folder/'record.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
