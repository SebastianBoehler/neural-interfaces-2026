"""Reauthenticate the original external factorial before examining all arm differences."""
import hashlib
import json
from pathlib import Path

import numpy as np

import score_f32x as original
from settings import sha, stamp

SEAL = "59efb4863bc6e9e93d075b4ddb492be641f9ed7fd4ac5f97c4ce0137ec68bea4"
SCORE = "90880bdbc01f62e3264edd2b8c87e7f3ec3e4f873b53bb22bacbec5c16a29958"


def run(out):
    prior = Path("before_labels.json")
    assert sha(prior)==SEAL and sha("result.json")==SCORE
    signed = json.loads(prior.read_text())

    def verify_existing(path, data):
        assert path == prior
        assert json.dumps(data,sort_keys=True)==json.dumps(signed,sort_keys=True)
        return SEAL

    save = original.seal
    original.seal = verify_existing
    bank,subject,visit,interface,streams,people,proof,_ = original.qualify()
    assert proof==SEAL and stamp(bank)==signed["predictions"]
    assert len(original.ARMS)==8 and original.ARMS[:4]==("stock","hard","single","multi")
    before = save(out/"before_labels.json",dict(held_labels_opened=False, parent=SEAL,
                  data=stamp(bank), source=hashlib.file_digest(Path(__file__).open("rb"),"sha256").hexdigest()))
    with np.load(original.DATA,allow_pickle=False) as z:
        y=z["y"]
    cells,classes=original.official_cells(bank,y,subject,visit,dict(stream=streams,interface=interface),people)
    rng=np.random.default_rng(2202)
    i=rng.integers(9,size=(16000,9,1,1))
    v=rng.integers(2,size=(16000,9,2,1))
    o=rng.integers(2,size=(16000,9,2,2))
    changes=cells[1:]-cells[:1]
    point=changes.mean((1,2,3))
    boot=changes[:,i,v,o].mean((2,3,4))
    scale=boot.std(1,ddof=1)
    standard=np.abs(boot-point[:,None])/scale[:,None]
    width=float(np.percentile(standard.max(0),80))*scale
    data={}
    for k,arm in enumerate(original.ARMS[1:]):
        data[arm]=dict(pp=float(point[k]), paired80=np.percentile(boot[k],[10,90]).tolist(),
                       simultaneous80=[float(point[k]-width[k]),float(point[k]+width[k])],
                       order_pp=changes[k].mean((0,1)).tolist())
    base= cells[2]-cells[1]
    conventional=dict(pp=float(base.mean()),paired80=np.percentile(base[i,v,o].mean((1,2,3)),[10,90]).tolist(),
                      order_pp=base.mean((0,1)).tolist(),
                      class_pp=(classes[2]-classes[1]).mean((0,1,2)).tolist(),
                      people_pp=dict(zip(people,base.mean((1,2)).tolist(),strict=True)))
    digest=save(out/"result.json",dict(before_labels=before,scope="retrospective complete-system external",
                                      all_vs_stock=data,conventional_vs_hard=conventional))
    print("PASS selection audit",digest,data,conventional,flush=True)


if __name__=="__main__":
    import sys
    run(Path(sys.argv[1]))
