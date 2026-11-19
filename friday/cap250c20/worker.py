"""Run the real corrected full-system consumers on authenticated F17 probabilities."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bridge_orders import write_npz, write_receipt
from bridge_owner import pin, sha, stamp
from complete_tail import count_filter, limit_session
from complete_zero import FEAT, HASHES
from outer import arms
from tails import personalize, scopes

WORK = Path(__file__).resolve().parent
PUPILS = WORK.parent/"fc20_pupils_v2"
F17 = PUPILS.parent.parent/"ni26_friday_s1c_phase_20261118"
F17_SOURCE = "7ed98f96ba6b05165a1bc035a20af0d7dc516b9b1d61c82533abcc85ecbf2372"
F17_BLIND = "77a25afbba7061dde2c781f3e8fb810b49d653aa57d09dcbc4b5abc0bb2abb37"
ORDERS_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
ARMS = ("stock","identity","hard","jitter")


def source_array(order, people, release):
    pin(F17/"actor.sha256",F17_SOURCE)
    pin(F17/"before_labels.sha256",F17_BLIND)
    folders = {}
    bank = {k:np.full((2,2655,3),np.nan) for k in ("hard","jitter")}
    for person in sorted(set(people)):
        rows = np.flatnonzero(people==person)
        for seed in (0,1):
            key=f"{person}_s{seed}.npz"
            path = PUPILS/"pupils"/key
            assert not path.stat().st_mode & 0o222
            folders[key]=sha(path)
            with np.load(path,allow_pickle=False) as f:
                ev=json.loads(str(f["metadata"]))
                assert ev["actor"]==F17_SOURCE and not ev["held_labels_opened"] and ev["source"]==release
                assert ev["person"]==person and ev["seed"]==seed and ev["order_sha"]==ORDERS_SHA
                assert ev["data"]==HASHES[FEAT] and ev["fold"]==sha(F17/"folds"/key)
                np.testing.assert_array_equal(f["index"],rows)
                for name,v in bank.items():
                    p, proof = f[f"{name}|{order}"], f[f"{name}|{order}|proof"]
                    assert p.shape==(len(rows),3) and proof.shape==(2,) and 0 <= proof[0] <= 2e-5
                    assert proof[1] >= 0 and np.isnan(v[seed,rows]).all()
                    v[seed,rows]=p
    assert len(folders)==20
    for v in bank.values():
        assert np.isfinite(v).all() and (v>=0).all()
        np.testing.assert_allclose(v.sum(2),1.,atol=1e-12,rtol=0)
    assert not np.array_equal(bank["hard"],bank["jitter"])
    return dict(identity=None, **{k:v.mean(0) for k,v in bank.items()}),folders


def run_tail(ctx, order, raw, selected, verification):
    count, final, windows = (ctx[k] for k in ("count","final","windows"))
    rows = {s:count.C.order_rows(order,s,count.PROTO) for s in selected}
    start = perf_counter()
    with scopes(count,selected):
        filtered, latch = count_filter(count,raw,order)
    count_time = perf_counter()-start
    start = perf_counter()
    work, scores = {},filtered.copy()
    for s,ix in rows.items():
        scores[ix], work[str(s)] = personalize(final, windows, raw, filtered, ix)
    csp_time = perf_counter()-start
    for s in verification:
        with limit_session(count,s):
            prefix,_ = count_filter(count,raw,order)
            ix = count.C.order_rows(order,s,count.PROTO)
        np.testing.assert_array_equal(ix,rows[s][:120])
        np.testing.assert_array_equal(prefix[ix],filtered[ix])
        observed, proof = personalize(final,windows,raw,prefix,ix)
        np.testing.assert_array_equal(observed,scores[ix])
        assert proof["refits"]==5 and proof["fusions"]==30
    keep=np.concatenate(list(rows.values()))
    return dict(raw=raw,filtered=filtered,terminal=scores), dict(
        count_latch={str(k):v for k,v in latch.items()},
        count_effect=int(np.any(filtered[keep]!=raw[keep],1).sum()),
        count_s=count_time, csp_s=csp_time, csp=work,
        csp_refits=sum(v["refits"] for v in work.values()),csp_fusions=sum(v["fusions"] for v in work.values()),
        input_sha=stamp(raw[keep]))


def main(order, mode, pupil_source):
    started=perf_counter()
    pin(PUPILS/"source.sha256",pupil_source)
    source=sha(WORK/"consumer.sha256")
    smoke=mode=="smoke"
    if not smoke:
        witness=json.loads((WORK/"smoke.json").read_text())
        assert witness["passed"] and witness["source"]==source and witness["pupil_source"]==pupil_source
    pin(FEAT,HASHES[FEAT])
    with np.load(FEAT,allow_pickle=False) as d:
        people=[name.split("/")[-1] for name in d["subject"]]
        visits=d["session"]
    people=np.array(people)
    assert len(set(people))==10
    pupil,folds=source_array(order, people, pupil_source)
    active=(people=="P01") if smoke else np.ones(len(people),dtype=bool)
    with arms(order,pupil,active) as ctx:
        count=ctx["count"]
        sessions=tuple(count.SESS)
        assert len(sessions)==14
        rows={s:count.C.order_rows(order,s,count.PROTO) for s in sessions}
        p01=next(s for s,r in rows.items() if set(people[r])=={"P01"} and set(visits[r])=={"1"})
        latch=count.latch(order)
        peer=next(s for s in sessions if s!=p01 and latch[s] is not None and latch[s]<120)
        prefixes=(p01,peer,p01)
        selected=(p01,peer) if smoke else sessions
        index=np.concatenate([rows[s] for s in selected])
        if not smoke:
            np.testing.assert_array_equal(index,ctx["delivery"])
        paths, work = {}, {}
        for k in ARMS:
            paths[k], work[k] = run_tail(ctx,order,ctx["raw"][k],selected,prefixes)
        for k in paths["stock"]:
            np.testing.assert_array_equal(paths["stock"][k][index],ctx["reference"][k][index])
            np.testing.assert_array_equal(paths["identity"][k][index],paths["stock"][k][index])
            if smoke:
                for name in ("hard","jitter"):
                    np.testing.assert_array_equal(paths[name][k][rows[peer]],paths["stock"][k][rows[peer]])
        assert all(v["count_effect"] > 0 for v in work.values())
        raw_changes, score_move = {},{}
        for arm in ("hard","jitter"):
            raw_changes[arm]=int(np.count_nonzero(paths[arm]["raw"][index].argmax(1) != paths["stock"]["raw"][index].argmax(1)))
            score_move[arm]=float(np.abs(paths[arm]["terminal"][index]-paths["stock"]["terminal"][index]).max())
            assert score_move[arm]>1e-4
            assert any(work[arm]["csp"][str(s)]["target_sha"] != work["stock"]["csp"][str(s)]["target_sha"] for s in selected)
        assert any(work["hard"]["csp"][str(s)]["target_sha"] != work["jitter"]["csp"][str(s)]["target_sha"] for s in selected)
        assert not np.array_equal(paths["hard"]["terminal"][index],paths["jitter"]["terminal"][index])
        for arm in ARMS:
            if not smoke:
                assert (work[arm]["csp_refits"],work[arm]["csp_fusions"])==(126,1395)
        result=dict(passed=True,scope=mode,order=order,pupil_source=pupil_source,source=source,
                    parent=ctx["parent"],archive_zero=ctx["zero"],rows=190 if smoke else len(index),work=work,
                    actual_replay_rows=len(index),sessions=len(selected),folds=folds,held_labels_opened=False,
                    prefix=[str(s) for s in prefixes], shadow_fixed=True, new_csp_targets=True,
                    outer=ctx["proof"],stock_bitwise=True,native_identity=True,raw_changes=raw_changes,
                    maximum_terminal_move=score_move,seconds=perf_counter()-started)
        if not smoke:
            bank=dict(index=np.arange(len(index)),delivery=index,
                      **{f"{arm}|{k}":v for arm in ARMS for k,v in paths[arm].items()})
            result["output"]=write_npz(WORK/f"{order}.npz",bank)
        path=WORK/f"{'smoke' if smoke else order}.json"
        write_receipt(path,**result)
    print("PASS F-C20",order,mode,sha(path),raw_changes,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("order",choices=("full0","within0"))
    p.add_argument("mode",choices=("smoke","complete"))
    p.add_argument("pupil_source")
    a=p.parse_args()
    assert a.mode!="smoke" or a.order=="full0"
    main(a.order,a.mode,a.pupil_source)
