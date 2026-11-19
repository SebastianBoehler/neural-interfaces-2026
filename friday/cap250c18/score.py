"""Read gold only after the full corrected four-arm factorial is sealed."""

import json
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from tails import ARMS
from upstream import ORDER, PARENT_RECEIPT, ZERO, archive_receipt, input_delivery


KEYS = ("P0r", *ARMS)


def qualified():
    _, reference = archive_receipt()
    proof = Path("complete.json").resolve()
    receipt = json.loads(proof.read_text())
    smoke = json.loads(Path("smoke.json").read_text())
    assert smoke["passed"] and smoke["scope"] == "smoke" and not smoke["held_labels_opened"]
    assert receipt["passed"] and receipt["scope"] == "complete" and not receipt["held_labels_opened"]
    assert receipt["order"] == ORDER and receipt["rows"] == 2655 and set(receipt["arms"]) == set(ARMS)
    assert receipt["parent"] == PARENT_RECEIPT and receipt["historical_zero"] == ZERO
    assert smoke["source"] == receipt["source"] == sha(Path("source.sha256"))
    assert receipt["stock_bitwise"] and receipt["matched_p0r"] and receipt["off_zero_exact"]
    assert receipt["independent_count"] and receipt["first_20_exact"] == 280
    assert receipt["max_raw_ft_effect"] > .001
    for name, (ft, cp) in ARMS.items():
        v = receipt["arms"][name]
        assert v["count_windows"] == 2655 and min(v["prefix_count"]) > 0
        assert v["ft_steps"] == 9600*int(ft) and v["warm_consumed"] == ft
        assert v["csp_refits"] == 126*int(cp) and v["csp_fusions"] == 1395*int(cp)
        assert v["csp_features"] == 2655*int(cp) and v["csp_constructed"] == cp
    path = Path("complete.npz").resolve()
    assert receipt["output"]["path"] == str(path)
    pin(path,receipt["output"]["sha256"])
    assert not (path.stat().st_mode & 0o222 or proof.stat().st_mode & 0o222)
    with np.load(path,allow_pickle=False) as banks, np.load(reference,allow_pickle=False) as stock:
        np.testing.assert_array_equal(banks["index"],np.arange(2655))
        np.testing.assert_array_equal(banks["delivery"],input_delivery())
        np.testing.assert_array_equal(banks["p0r"],stock["corrected|P0r_posterior"])
        for key in ("raw","filtered","terminal"):
            np.testing.assert_array_equal(banks[f"stock|{key}"],stock[f"corrected|cap250_{key}"])
        for name in ("no_csp","both_off"):
            np.testing.assert_array_equal(banks[f"{name}|filtered"],banks[f"{name}|terminal"])
        p = np.stack([banks["p0r"],*[banks[f"{name}|terminal"] for name in ARMS]])
        assert p.shape == (5,2655,3) and np.isfinite(p).all()
    return p.argmax(-1), proof, receipt


def main():
    estimates, proof, upstream = qualified()
    pin(METRIC,METRIC_HASH)
    pin(FEAT,HASHES[FEAT])
    metric = start_module("f18_official",METRIC,METRIC.read_text())
    with np.load(FEAT,allow_pickle=False) as z:
        true, subject, visit, interface = (z[key] for key in ("y","subject","session","interface"))
    names = sorted(set(subject))
    groups = np.stack((subject,visit,interface),axis=1)
    sums, weights, person = [], [], {}
    for who in names:
        m = subject == who
        cells = np.column_stack([metric.group_scores(metric.balanced_accuracy,true[m],pred[m],groups[m])
                                 for pred in estimates])
        sums.append(cells.sum(0))
        weights.append(len(cells))
        person[who] = 100*cells.mean(0)
    sums, weights = np.asarray(sums), np.asarray(weights)
    assert len(names)==10 and weights.sum()==28
    draw = np.random.default_rng(460).integers(0,len(names),(4000,len(names)))
    point = 100*sums.sum(0)/weights.sum()
    boot = 100*sums[draw].sum(1)/weights[draw].sum(1)[:,None]
    np.testing.assert_allclose(point[:2],[75.7784971844035,78.3241479921539],atol=1e-4,rtol=0)
    identity = {k: np.eye(5)[j] for j,k in enumerate(KEYS)}
    p0,s,f,c,b = (identity[key] for key in KEYS)
    definitions = dict(stock_minus_p0r=s-p0, no_ft_minus_p0r=f-p0, no_csp_minus_p0r=c-p0,
                       both_off_minus_p0r=b-p0, ft_given_csp=s-f, ft_without_csp=c-b,
                       csp_given_ft=s-c, csp_without_ft=f-b, interaction=s-f-c+b,
                       stock_minus_both=s-b)
    summaries = {}
    for key, vector in definitions.items():
        ci = np.percentile(boot@vector,(10,90)).tolist()
        summaries[key] = dict(pp=float(point@vector), paired80=ci,
                              per_person_pp={name:float(p@vector) for name,p in person.items()},
                              outcome="positive" if ci[0]>0 else "nonpositive" if ci[1]<=0 else "undecided")
    cheaper = {name: summaries[key]["paired80"][1] <= .30 for name,key in (
        ("no_ft","ft_given_csp"),("no_csp","csp_given_ft"),("both_off","stock_minus_both"))}
    names = {name:dict(zip(KEYS,values.tolist(),strict=True)) for name,values in person.items()}
    choices = dict(zip(KEYS,point.tolist(),strict=True))
    result = Path.cwd()/"score.json"
    write_receipt(result,scope="FULL-SYSTEM DEV on D-EAwcf corrected held-new within0 only",
                  order=ORDER, official_cells=28, proof_sha=sha(proof), scoring=sha(Path("scoring.sha256")),
                  absolute_pct=choices, contrasts=summaries, people_pct=names,
                  paired_person_bootstrap=dict(draws=4000,seed=460,interval=.80),
                  work=upstream["arms"], comparable_for_another_order=cheaper,
                  forecasts_pp=dict(ft_given_csp=1.10,csp_given_ft=.30,stock_minus_both=1.20,interaction=.20))
    print("PASS F-C18 factorial score",sha(result),flush=True)
    print(json.dumps(dict(absolute_pct=choices,contrasts=summaries,comparable=cheaper),sort_keys=True),flush=True)


if __name__=="__main__":
    main()
