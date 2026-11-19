"""Run the archived Solver factories on two untouched passes and the hard-only trial."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from fixture import chosen
from qualify import ARM_ORDER, fixtures, load, measured, proof, protect, resets
from settings import sha, stamp
from build_hard import ACTORS, LOCK, STOCK

TRIAL = "6a569180d343c038558fe7eccb7903a36e2bdf694af0df3687bff3b17049fbcc"
DIAG = "ba3cc6f6d9bdf373ea065e76c293b2b7bee46b281ca729d71a12259e24e9a190"
P10 = "e1b650082d00b0577453a4c4abb7e378b6993309d0c199c324878ba355d1d808"


def expected_cpu(data, rows):
    actors = []
    for seed, digest in enumerate(ACTORS):
        path = Path("/actors/hard")/f"raw_s1c_s{seed}.pt"
        assert sha(path) == digest
        actors.append(torch.jit.load(str(path),map_location="cpu").eval())
    out = []
    with torch.no_grad(),torch.jit.optimized_execution(False):
        assert all(a.reset().item()==0 for a in actors)
        for row in rows:
            x = data[int(row):int(row)+1].contiguous()
            emit = [a(x) for a in actors]
            assert all(value.shape==(1,3) and value.dtype==torch.float64 and value.device.type=="cpu" for value in emit)
            out.append(np.stack([value.numpy()[0] for value in emit]))
    result=np.asarray(out)
    assert np.isfinite(result).all() and (result>=0).all()
    np.testing.assert_allclose(result.sum(-1),1.,rtol=0,atol=1e-12)
    return result


def assess(stock, trial, x, expected, reverse):
    if reverse:
        b = measured(stock,x)
        m = measured(trial,x,expected)
        a = measured(stock,x)
    else:
        a = measured(stock,x)
        m = measured(trial,x,expected)
        b = measured(stock,x)
    for run in (a,m,b):
        work=run[1]["work"]
        assert work["ft_over"]==work["errors"]==0 and work["windows"]==len(x)
        assert work["ft_steps"]==20*work["ft_refits"]
    derived=proof(stock,m[1],m[3])
    assert derived["calls"]==len(x)==len(m[3].pairs)
    aa=protect(a,b,None)
    ab=protect(a,m,None)
    ba=protect(b,m,None)
    work=[pair[1]["work"] for pair in (a,b,m)]
    positions=[work[0]["recognized"],work[1]["recognized"],work[2]["recognized"]]
    assert len(set(positions))==1
    if positions[0]<0:
        assert derived["warm_rows"]>0 and derived["csp_fusions"]>0
    labels=[p[1]["labels"].tolist() for p in (a,b,m)]
    return dict(labels=labels,work=work,proof=derived,reversed=reverse,
                stock_AA=dict(terminal_max=float(np.abs(a[1]["stages"][:,2]-b[1]["stages"][:,2]).max()),**aa),
                max_protected_trial=[max(x["max"] for x in r["stages"].values()) for r in (ab,ba)]),m[1]


def run(mode,index,stock_zip,trial_zip):
    assert torch.__version__=="2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    assert sha(stock_zip)==STOCK and sha(trial_zip)==TRIAL
    assert sha(Path("/diagnosis.json"))==DIAG
    assert sha(Path("/p10_smoke.json"))==P10
    cases,_,_=fixtures()
    sessions,(eeg,cold,anchor,archive),channels,who,_=chosen(mode,index)
    result={}
    source=sha("source.sha256")
    if mode=="produce":
        s=json.loads(Path("smoke.json").read_text())
        assert s["passed"] and s["source"]==source and s["archive"]==TRIAL and not s["held_labels_opened"]
    with TemporaryDirectory() as temp:
        stock,trial=load(stock_zip,trial_zip,Path(temp),channels)
        gx,_,gq,gold,_,prior=cases[ARM_ORDER[0]]
        _, base, _, _ = measured(stock,gx)
        assert base["labels"].tolist()==gold.tolist() and base["work"]==prior["activity"]
        _, setup, _, actors = measured(trial,gx,gq)
        assert len(actors.pairs)==len(gx) and setup["work"]["errors"]==0
        _, obs, _, _=measured(stock,eeg[cold].to("cuda:0"))
        assert obs["labels"].tolist()==anchor["labels"] and obs["work"]==anchor["activity"]
        for position,(key,(data,rows)) in enumerate(sessions.items()):
            x=data[rows].to("cuda:0")
            expected=expected_cpu(data,rows)
            r,live=assess(stock,trial,x,expected,position in (1,2))
            r["rows"]=rows.tolist()
            r["waveform"]=stamp(data[rows].numpy())
            if mode=="smoke":
                _, old_rows, cpu, archived_stock, archived_hard, _ = cases[key]
                np.testing.assert_array_equal(rows,old_rows)
                np.testing.assert_array_equal(expected,cpu)
                assert r["labels"]==[archived_stock.tolist()]*2+[archived_hard.tolist()]
                peer=cases[ARM_ORDER[1-ARM_ORDER.index(key)]][0][:1]
                r["reset"]=resets(trial,peer,x,expected,live)
            result[key]=r
            print("PASS actual archive streams",key,len(rows),flush=True)
    out=Path("smoke.json" if mode=="smoke" else f"person_{index}.json")
    with out.open("x") as fp:
        json.dump(dict(passed=True,mode=mode,index=index,who=who,source=source,archive=TRIAL,stock=STOCK,
                       actors=LOCK,cold_archive=archive,cold_rows=cold.tolist(),cold_work=obs["work"],
                       cold_labels=obs["labels"].tolist(),worker=torch.cuda.get_device_name(),
                       held_labels_opened=False,results=result),fp,sort_keys=True,allow_nan=False)
        fp.write("\n")
    out.chmod(0o444)
    print("PASS complete actual archive",sha(out),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","produce"))
    p.add_argument("index",type=int,choices=range(9))
    p.add_argument("stock",type=Path)
    p.add_argument("trial",type=Path)
    a=p.parse_args()
    run(a.mode,a.index,a.stock,a.trial)
