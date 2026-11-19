"""Measure RTX repeat variation in the untouched cap250 path."""

import json
from pathlib import Path

import numpy as np
import torch

import replay
from settings import UPPER, schedule, sha
from worker import authentic_stock, stream


def main():
    assert "RTX 2080 Ti" in torch.cuda.get_device_name() and torch.__version__=="2.14.1+cu130"
    with np.load("/external.npz",allow_pickle=False) as z:
        eeg=torch.from_numpy(z["X"])
        subject, days, channels=z["subject"],z["session"],z["ch_names"]
    who="Scherer2015/E|1"
    plans=schedule(subject,days)
    rows=plans[9411][who]
    archived=json.loads(Path("/rtx_other/rtx_cap_3.json").read_text())["sessions"][f"{who}|0"]
    assert len(rows)==120 and archived["rows"]==plans[0][who].tolist()
    model=replay.load(Path("/package"),channels.tolist())
    cold=stream(model,(),eeg,plans[0][who],"stock")
    authentic_stock(cold,archived)
    runs=[stream(model,(),eeg,rows,"stock") for _ in range(3)]
    entries=[]
    for i,j in ((0,1),(0,2),(1,2)):
        a,b=(runs[k][1] for k in (i,j))
        fit=a["loss"]==b["loss"]
        acts=runs[i][0]["activity"]==runs[j][0]["activity"]
        delta=np.abs(a["upstream"]-b["upstream"]).max(axis=(0,2))
        changed=int(np.count_nonzero(a["terminal"].argmax(1)!=b["terminal"].argmax(1)))
        assert fit and acts and np.array_equal(a["upstream"][:,:4],b["upstream"][:,:4])
        entries.append(dict(pair=[i,j],upstream=delta.tolist(),loss_exact=fit,counts_exact=acts,
                            terminal_changes=changed,terminal_max=float(np.abs(a["terminal"]-b["terminal"]).max())))
    result=dict(archive=sha("/rtx_other/rtx_cap_3.json"),source=sha("source.sha256"),
                rows=rows.tolist(),comparisons=entries,exceeds_prior_bound=any(x["upstream"][-1]>UPPER for x in entries),
                held_labels_opened=False,stock_only=True,times=[r[0]["seconds"] for r in runs])
    with Path("stock_repeat.json").open("x") as f:
        json.dump(result,f,sort_keys=True,allow_nan=False)
        f.write("\n")
    Path("stock_repeat.json").chmod(0o444)
    print("PASS F-C20e pure stock repeat", sha("stock_repeat.json"), result["comparisons"], flush=True)


if __name__=="__main__":
    main()
