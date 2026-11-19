"""Read the sealed source tape, original F32e models and fixed source training rows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, row_sha, sha
from bounded_bn import normalize
from encoder312 import fingerprint
import model322
from source_labels import read_training_labels
from source_targets import normalizer


WORK = Path(__file__).resolve().parent
BASE = WORK.parent
TAPE = BASE/"ni26_friday_c31c_source_mix_v1_20261119"
INPUT = BASE/"ni26_s323_normsplit_20261117/data/proteus_windows.npz"
HARD = BASE/"ni26_friday_s1c_motor_external_cpu_20261119"
SOURCE = "03bbd2e27e0b493f02c63099c0f0d1d57a8c802b65f78203f7f04a8461f16136"
SEAL = "73c4378ab5dafcd9a3319fb9538203a9eee7903aebb6593e02bd07234933d180"
HARD_ACTOR = "fdbc95d3649c0a55efb1c4de6fdcfcb19b8d278a6c319392a7e3872f1d90417e"
HARD_DIGESTS = ("bcedeb0fb01b1735a834f9c8a7270d04f97fe085abce31a5c0bb05d21aa275f5",
                "82b9ae356bc5e09131f3b5db3ced8674f1691e1839031c78e020844df233781d")
FIELDS = ("stock_mix","frozen","warm","active")
SEED = 31035


def load():
    path,bank = TAPE/"source-tape.json",TAPE/"source-tape.npz"
    stamp = json.loads(path.read_text())
    if (not stamp["qualified"] or stamp["held_labels_opened"] or stamp["external_predictions_created"]
            or stamp["source"] != SOURCE or stamp["seal"] != SEAL or stamp["data"] != DATA_SHA
            or stamp["rows"] != 2655 or stamp["warm_rows"] != 2375 or stamp["bank"] != sha(bank)
            or stamp["operational"]["external_sessions"] != 36 or not stamp["operational"]["mathematical_scope_only"]
            or stamp["operational"]["held_labels_opened"] or sha(INPUT) != DATA_SHA):
        raise ValueError("guarded source tape or prior natural external workload did not qualify")
    with np.load(bank,allow_pickle=False) as a, np.load(INPUT,allow_pickle=False) as z:
        x,groups,visits,interface = (z[k] for k in ("X","subject","session","interface"))
        names = np.array([f"{s.rsplit('/',1)[-1]}-{v}" for s,v in zip(groups,visits,strict=True)])
        rows,order,weights,rest = (a[k] for k in ("index","delivery","weights","rest"))
        np.testing.assert_array_equal(a["session"],names)
        raw = [a["forced|"+k] for k in FIELDS]
        ages = a["forced|age"]
    np.testing.assert_array_equal(rows,np.arange(2655))
    assert x.shape==(2655,41,480) and x.dtype==np.float32 and len(set(names))==14 and len(set(groups))==10
    pos = [order[names[order]==n] for n in sorted(set(names))]
    for ix in pos:
        np.testing.assert_array_equal(ages[ix],np.arange(len(ix)))
        np.testing.assert_array_equal(raw[-1][ix],ages[ix]>=20)
    y = read_training_labels(INPUT,rows,len(x))
    assert np.isin(y,(0,1,2)).all()
    weight = np.full(len(x),np.nan)
    for ix in pos:
        for kind in np.unique(interface[ix]):
            cell = ix[interface[ix]==kind]
            counts = np.bincount(y[cell],minlength=3)
            if (counts==0).any():
                raise ValueError("source balanced loss encountered an incomplete interface cell")
            weight[cell] = 1/(3*len(set(interface[ix]))*counts[y[cell]])
        np.testing.assert_allclose(weight[ix].sum(),1.,atol=1e-12,rtol=0)
    rng = np.random.default_rng(SEED)
    schedule = np.concatenate([rng.permutation(len(pos)) for _ in range(8)])
    rng = np.random.default_rng(SEED)
    donor = np.full(len(x),-1,dtype=np.int64)
    for label in range(3):
        for warm in (False,True):
            ix = np.flatnonzero((y==label)&(raw[-1]==warm))
            shuffled = rng.permutation(ix)
            donor[shuffled] = np.roll(shuffled,1)
    if (donor<0).any() or (donor==rows).any() or not np.array_equal(y[donor],y) or not np.array_equal(raw[-1][donor],raw[-1]):
        raise ValueError("same-stratum complete context derangement failed")
    cross = float(np.mean(names[donor]!=names))
    if cross<=.5 or stamp["replaceable_nominal_mass"]!=5/96 or float(rest)!=.0625:
        raise ValueError("actual source shuffle matching or exact physical mass invalid")
    binding = dict(tape=sha(path),bank=sha(bank),ages=row_sha(ages),delivery=row_sha(order),
                   schedule=row_sha(schedule),nominal_donor=row_sha(donor),donor_cross_session=cross,
                   source_people=sorted(set(groups)),source_rows=row_sha(rows))
    return (torch.from_numpy(x).cuda(), torch.from_numpy(y.astype(np.int64)).cuda(), torch.from_numpy(weight).cuda(),
            [torch.from_numpy(a).cuda() for a in raw], torch.from_numpy(weights).cuda(), float(rest), pos,
            schedule, donor, binding)


def pupil(seed):
    old = HARD/"checkpoints"/f"hard_{seed}_all_source.pt"
    card = json.loads(old.with_suffix(".json").read_text())
    if (sha(HARD/"actor.sha256") != HARD_ACTOR or sha(old) != HARD_DIGESTS[seed]
            or card["checkpoint"] != HARD_DIGESTS[seed] or card["dataset"] != DATA_SHA or card["origin"] != ORIGIN_SHA
            or card["kind"] != "hard" or card["held"] != "all_source" or card["seed"] != seed
            or card["actual_aux"] or not card["auxiliary_excluded"] or card["held_labels_opened"]):
        raise ValueError("original CPU-qualified hard-pair source state differs")
    pl = torch.load(old,map_location="cpu",weights_only=True)
    if any(pl[k]!=card[k] for k in ("kind","held","seed","scale")):
        raise ValueError("pinned hard fit metadata differs")
    model322.SEED = seed
    model = model322.Factor("session",False,"cuda")
    model.load_state_dict(pl["state"],strict=True)
    model.eval()
    if fingerprint(model) != card["state"]:
        raise ValueError("native F32e pretrained pupil tensors differ")
    layer = normalizer(model)
    moments = tuple(a.detach().clone() for a in (layer.running_mean,layer.running_var))
    layer.forward = lambda a: normalize(layer,a,[])
    return model,card["scale"],layer,moments,card["state"]


def rng_digest():
    raw = (torch.random.get_rng_state(),torch.cuda.get_rng_state())
    return hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in raw)).hexdigest()
