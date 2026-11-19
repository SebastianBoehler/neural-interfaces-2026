"""Exact native source identities and read-only spatial prior artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np

DATA_SHA="2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
ACTOR_SHA="37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f"
WEIGHTS_SHA="7cf13696190983cd454880d6ecb8c049902a12ec694c1a24ff74cd4013a688a2"
PEOPLE=tuple(f"P{i:02d}" for i in range(1,11))
NAMES=("ALL",*PEOPLE)
EXTERNAL_ARMS=("stock","identity","transport","reflection","pooled")
PUBLIC_ARMS=(*EXTERNAL_ARMS,"hard","transport_hard")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def stamp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(path,data):
    path=Path(path)
    with path.open("x") as f:
        json.dump(data,f,sort_keys=True,allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    return sha(path)


def prior(name):
    assert name in NAMES
    p=Path("source.json")
    r=json.loads(p.read_text())
    assert r["data"]==DATA_SHA and r["actor"]==ACTOR_SHA and r["weights"]==WEIGHTS_SHA
    assert r["source_fit"]==sha("source-fit.sha256") and not r["scherer_access"] and set(r["priors"])==set(NAMES)
    smoke=Path("fit_smoke.json")
    assert r["smoke"]==sha(smoke) and not (p.stat().st_mode|smoke.stat().st_mode)&0o222
    path=Path(f"prior_{name}.npz")
    assert sha(path)==r["priors"][name] and not path.stat().st_mode&0o222
    with np.load(path,allow_pickle=False) as z:
        x,index=z["prior"],z["source_index"]
        m=json.loads(str(z["metadata"]))
    assert x.shape==(8,3,41,41) and m["name"]==name and m["source_rows"]==len(index)
    assert m["source_rows_hash"]==stamp(index) and m["matrix"]==stamp(x)
    assert m["data"]==DATA_SHA and m["actor"]==ACTOR_SHA and m["weights"]==WEIGHTS_SHA
    assert not m["scherer_access"] and m["source_persons"]==[p for p in PEOPLE if p!=name]
    assert len(index)==len(set(index)) and (index[1:]>index[:-1]).all()
    assert np.isfinite(x).all()
    np.testing.assert_allclose(x.mean(1),np.broadcast_to(np.eye(41),(8,41,41)),rtol=0,atol=2e-11)
    return x,m,sha(path)
