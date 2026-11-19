"""Authenticate and reconstruct all six held-person §304 streams."""

import json
from pathlib import Path

import numpy as np

from bridge_owner import BASE, ROOT, pin, stamp
from complete_zero import FEAT, HASHES
from member_inputs import ORDERS, pinned_delivery

C30=BASE/"ni26_s304_coral_prod02_20261116"
PROOF=ROOT/"runs/coral304_inputs"
SOURCES="94a5f8a19c3532017f42fc1e85520e9ca0aa511e2ece1bc40df9f406a65b45aa"
ASSEMBLED="39f012c9196cb8898d0c24841f8af60d55b3d3b8c5f87c52ea67c28abe1824bf"
VALIDATED="0c020596c7ee06b895fbced6f6c4efa5ffbb88b9a77a8a2f260b90f11f8ab070"
LOCK="74ac1a8056a21d58cf52311ad9a4cd0c76636f9278daa84092fa6d1e5179af3b"
PERM="6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"


def read(path):
    return json.loads(path.read_text())


def contents(order):
    pin(C30/"coral304.sha256",SOURCES)
    pin(PROOF/"streams.sha256",ASSEMBLED)
    pin(PROOF/"validation.json",VALIDATED)
    pin(PROOF/"lock.json",LOCK)
    pin(FEAT,HASHES[FEAT])
    source=read(PROOF/"validation.json")
    lock=read(PROOF/"lock.json")
    assert lock==dict(assembler_validation_sha256=VALIDATED,manifest_sha256=ASSEMBLED,
                      producer_manifest_sha256=SOURCES,source=str(C30))
    assert source["producer_manifest_sha256"]==SOURCES and not source["held_accuracy_computed"]
    assert source["data_sha256"]==HASHES[FEAT] and source["permutation_sha256"]==PERM
    assert source["counts"]==dict(folds=240,streams=192)
    manifest={name:digest for digest,name in (line.split() for line in (PROOF/"streams.sha256").read_text().splitlines())}
    assert manifest==source["stream_hashes"]
    with np.load(FEAT,allow_pickle=False) as z:
        subjects=z["subject"]
    roster=sorted(set(subjects))
    assert len(roster)==10
    p=pinned_delivery(ROOT/"runs/shuffle_perms.npz")[ORDERS.index(order)]
    arrays=np.full((6,2655,3),np.nan)
    fold_hash={}
    streams={}
    for seed in range(6):
        for subject in roster:
            who=subject.split("/")[-1]
            name=f"conditional30_s{seed}_{who}.npz"
            digest=source["producer_hashes"][name]
            artifact=C30/"runs/coral304"/name
            pin(artifact,digest)
            fold_hash[name]=digest
            fit=np.flatnonzero(subjects!=subject)
            held=p[subjects[p]==subject]
            with np.load(artifact,allow_pickle=False) as z:
                info=json.loads(str(z["metadata"]))
                expected=dict(arm="conditional30",seed=seed,held=who,fit_people=[s for s in roster if s!=subject],
                          train_rows_sha256=stamp(fit.astype("<i8")),manifest_sha256=SOURCES,
                          data_sha256=HASHES[FEAT],permutation_sha256=PERM)
                assert all(info[k]==v for k,v in expected.items()) and len(info["parameter_sha256"])==64
                reg=info["penalty"]
                assert reg["batches"]>0 and min(reg["class_pairs"])>0 and reg["gradient_norm"]>0
                assert np.isfinite(reg["gradient_norm"]) and info["dg_stats"]["coral_pairs"]>0
                assert info["dg_stats"]["coral_pen"]>0
                np.testing.assert_array_equal(z[f"{order}|index"],held)
                prob=z[f"{order}|proba"]
                assert prob.shape==(len(held),3) and np.isfinite(prob).all() and (prob>=0).all()
                np.testing.assert_allclose(prob.sum(1),1.,rtol=0,atol=1e-12)
                assert np.isnan(arrays[seed,held]).all()
                arrays[seed,held]=prob
        name=f"ttt18{order}_fb_cc30_base-all_loso-s{seed}.npz"
        target=ROOT/"runs/diag/ttt/probs18"/name
        pin(target,manifest[name])
        streams[name]=manifest[name]
        with np.load(target,allow_pickle=False) as z:
            np.testing.assert_array_equal(z["index"],np.arange(2655))
            np.testing.assert_array_equal(arrays[seed],z["proba"])
    assert len(fold_hash)==60 and len(streams)==6 and np.isfinite(arrays).all()
    mean=np.clip(arrays.mean(0),1e-6,None)
    mean/=mean.sum(1,keepdims=True)
    return mean,dict(folds=fold_hash,streams=streams,pinned_order=stamp(p),joint_members=6)
