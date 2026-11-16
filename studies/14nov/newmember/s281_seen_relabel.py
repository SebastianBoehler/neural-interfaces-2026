"""Produce seen-session r15 prefixes from the inner v5.1b shadow.

Usage: python -I studies/14nov/newmember/s281_seen_relabel.py ORDER BASE
Requires the matched recognition trace from s281_seen_route.py. No treatment
or accuracy is scored. Prefixes at or after first recognition are discarded.
Each base has its own shadow and recognition trace. The earlier all-LOSO
study reused B2_pool labels across bases; that is an approximation.
"""
import ast
import hashlib
import importlib
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
BASES = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")
PACKAGE_SHA = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
PERMS_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
CAP = 250  # the package permits a refit at t=250; the next one is skipped


def require_matching_base(trace, base):
    assert base in BASES and f"{base}|shadow" in trace and f"{base}|recognised" in trace, "matching base missing from route trace"


def require_matching_artifact(artifact, order, base):
    """A consumer must call this before using an explicitly base-indexed prefix."""
    assert order in ORDERS and base in BASES, "invalid requested order or base"
    assert "base" in artifact and str(artifact["base"]) == base, "prefix base mismatch"
    assert "order" in artifact and str(artifact["order"]) == order, "prefix order mismatch"


def validate_prefix(prefix, pos, sessions, grid, recognised):
    assert prefix.shape == (len(pos), len(grid), 3) and len(sessions) == len(pos) == len(recognised)
    assert recognised.dtype == bool and np.array_equal(grid, np.arange(20, 20 + 10 * len(grid), 10))
    finite = np.isfinite(prefix)
    assert np.all(finite.all(2) | ~finite.any(2)), "partial or nonfinite prefix probability"
    for session in np.unique(sessions):
        rows = np.flatnonzero(sessions == session); rows = rows[np.argsort(pos[rows])]
        assert np.array_equal(pos[rows], np.arange(len(rows))), "session positions are not a permutation"
        r = recognised[rows]; assert np.all(r[1:] >= r[:-1]), "recognition unlatched within a session"
        for k, t in enumerate(grid):
            valid = finite[rows, k].all(1)
            assert not valid[t:].any(), "future window present in prefix"
            if t > CAP:
                assert not valid.any(), "post-cap prefix must not be consumed"
            elif t < len(rows) and not r[t]:
                assert valid[:t].all(), "missing pre-recognition prefix label"
                assert np.allclose(prefix[rows[:t], k].sum(1), 1), "prefix is not a posterior"
            else:
                assert not valid.any(), "post-recognition prefix must not be consumed"


def _ast_class(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name]
    assert len(nodes) == 1, f"expected exactly one {name} class"
    return ast.dump(nodes[0], include_attributes=False)


def package_stages():
    package = ROOT / "runs/sealed/final_zips/v66_cp1_cap250.zip"
    assert hashlib.sha256(package.read_bytes()).hexdigest() == PACKAGE_SHA, "v6.6 ZIP SHA differs"
    with zipfile.ZipFile(package) as z:
        bundle = z.read("submission.py").decode()
        with np.load(io.BytesIO(z.read("weights.npz"))) as weights:
            config = {k: weights[k] for k in weights.files}
    stage_source = (ROOT / "lean/src66/stages.py").read_text()
    for name in ("SessionEM", "TSEM"):
        assert _ast_class(bundle, name) == _ast_class(stage_source, name), f"package {name} class differs"
    expected = {"v6_ft_relabel": 1, "v6_routed": 1, "v6_ft_cap": CAP, "v6_ft_labels": 0,
                "v6_ft_t0": 20, "v6_ft_every": 10, "v6_ft_steps": 5,
                "v6_ft_lr": 0.001, "v6_ft_lam": 1.0, "v6_ft_seed": 0, "v6_ft_on": 1}
    for key, value in expected.items():
        assert config[key].item() == value, f"unexpected package setting {key}"
    sys.path.insert(0, str(ROOT))
    return importlib.import_module("lean.src66.stages"), config


def compare_package_stages(b, features, pc, em_out, out, prefix, pos, sessions, recognised, grid):
    stages, config = package_stages(); count = 0; max_out = max_prefix = 0.0
    for session in np.unique(sessions):
        rows = np.flatnonzero(sessions == session); rows = rows[np.argsort(pos[rows])]
        em, ts = stages.SessionEM(config), stages.TSEM(config)
        for t, row in enumerate(rows):
            if recognised[row]:
                break
            em.features = lambda _x, f=features[row]: f
            q = em.step(None, b[row]); pred = ts.step(pc[row], q, em.gate)
            max_out = max(max_out, float(np.max(np.abs(q - em_out[row]))), float(np.max(np.abs(pred - out[row]))))
            if t in grid and t <= CAP:
                k = int((t - 20) // 10); replay = ts.reapply(t, em.reapply(t))
                max_prefix = max(max_prefix, float(np.max(np.abs(replay - prefix[rows[:t], k])))); count += 1
    assert count > 0, "no pre-recognition refit points were checked"
    assert max_out < 1e-8 and max_prefix < 1e-8, f"package/study reapply mismatch: {max_out}, {max_prefix}"
    return dict(refits=count, max_output_difference=max_out, max_prefix_difference=max_prefix)


def check_causality(S, chain, scope, p, base, sessions, pos, original_out, original_prefix):
    """Perturb later windows and other sessions; compare the protected prefix."""
    original_apply = S.apply
    target = np.unique(sessions)[0]
    trows = np.flatnonzero(sessions == target); cutoff = 20
    assert len(trows) > cutoff
    reports = {}
    try:
        for case in ("future", "other_sessions"):
            def apply_with_perturbation(P):
                original_apply(P)
                z = S.A.L.Z
                mask = ((sessions == target) & (pos >= cutoff)) if case == "future" else (sessions != target)
                for key in ("GE|seen_session", "pc128|seen_session"):
                    value = z[key].copy(); n = int(mask.sum())
                    change = np.sin(np.arange(n)[:, None] * 1.3) * np.linspace(0.5, 2.0, value.shape[1])[None]
                    value[mask] += change; z[key] = value
            S.apply = apply_with_perturbation
            scope["CALLS"].clear()
            chain["chain"](base, "P0r", p)
            changed_out, changed_prefix = scope["CALLS"][-1][3:5]
            changed_region = ((sessions == target) & (pos >= cutoff)) if case == "future" else (sessions != target)
            moved = float(np.max(np.abs(changed_out[changed_region] - original_out[changed_region])))
            assert moved > 1e-8, f"{case}: perturbation did not change any output"
            if case == "future":
                rows = trows[pos[trows] < cutoff]
                assert np.allclose(changed_out[rows], original_out[rows], atol=1e-10, rtol=0)
                assert np.allclose(changed_prefix[rows, 0], original_prefix[rows, 0], atol=1e-10, rtol=0)
            else:
                assert np.allclose(changed_out[trows], original_out[trows], atol=1e-10, rtol=0)
                assert np.allclose(changed_prefix[trows], original_prefix[trows], atol=1e-10, rtol=0, equal_nan=True)
            reports[case] = moved
    finally:
        S.apply = original_apply
    return reports


def produce(order, base):
    assert order in ORDERS and base in BASES, "order and base must be explicit and registered"
    for key in ("SCHEME", "PSEED", "PROTO", "LABEL_OUT", "LABEL_ARM", "T17_DIR", "FT_TAG", "OCU_DIR", "RS_DIR", "T6_DIR", "FT_V", "REC_PFX"):
        assert key not in os.environ, f"unexpected input override: {key}"
    sc = order.rstrip("0123456789"); sd = order[len(sc):]
    os.environ.update(SCHEME=sc, PSEED=sd, PROTO="seen_session", KIND="v51b", STRICT="1", GATE_P0="1",
                      SCHEMES=sc, PERMS=str(int(sd) + 1), FT_V="s15", FT_TAG=f"ttt18{order}QF",
                      REC_PFX="ttt7", T6_DIR=str(ROOT / "runs/ttt/probs7"), T17_DIR=str(ROOT / "runs/ttt/probs18"),
                      OCU_DIR=f"runs/ocushuf_ss/{order}", RS_DIR=str(ROOT / "runs/restshuf"))
    relabel_path = ROOT / "studies/14nov/ftrelabel/relabel.py"
    relabel = relabel_path.read_text().split("\ngm.em_g, gm.tsem_g = em_hook, ts_hook")[0]
    assert relabel.count('if proto != "all_loso":') == 2
    relabel = relabel.replace('if proto != "all_loso":', 'if proto != "seen_session":')
    scope = {"__name__": "seen_relabel_lib", "__file__": str(relabel_path)}
    exec(compile(relabel, str(relabel_path), "exec"), scope)
    S, gm = scope["S"], scope["gm"]
    assert hashlib.sha256((ROOT / "runs/shuffle_perms.npz").read_bytes()).hexdigest() == PERMS_SHA
    gm.em_g, gm.tsem_g = scope["em_hook"], scope["ts_hook"]
    path = ROOT / "studies/13nov/shufcombo.py"
    source = path.read_text().split("\nident = np.arange(N); R = {}; Pperm")[0]
    old = "    return SC.person(out, p0)"
    assert source.count(old) == 1, "unexpected chain return"
    source = source.replace(old, "    return out[np.argsort(P)].copy()")
    chain = {"__name__": "seen_shadow_lib", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), chain)
    assert chain["p0"] == "seen_session"
    p = S.perm_for(sc, int(sd))["seen_session"]; idx = chain["IDX"]
    pinned = np.load(ROOT / "runs/shuffle_perms.npz")[f"{sc}|{sd}|seen_session"]
    assert np.array_equal(idx[p], pinned), "member and stage delivery orders differ"
    emitted = chain["chain"](base, "P0r", p)
    calls = scope["CALLS"]; assert len(calls) == 2 and calls[0][0] == "em" and calls[1][0] == "ts"
    _, _, _, out, prefix, pos, _ = calls[-1]
    b, em_out = calls[0][1].copy(), calls[0][3].copy()
    assert np.array_equal(out[np.argsort(p)], emitted), "captured label chain differs from P0r"
    trace_path = ROOT / f"runs/s281_seen_route/{order}.npz"
    with np.load(trace_path) as trace:
        assert np.array_equal(trace["index"], idx) and str(trace["order"]) == order
        require_matching_base(trace, base)
        recognised = trace[f"{base}|recognised"]
        shadow = trace[f"{base}|shadow"]
    pre = ~recognised
    assert np.allclose(emitted[pre], shadow[pre], atol=1e-9, rtol=0), "inner shadow differs from route trace"
    sessions = S.A.L.Z["sess|seen_session"].copy()
    features = S.A.L.Z["GE|seen_session"].copy(); pc = S.A.L.Z["pc128|seen_session"].copy()
    grid = np.arange(20, 20 + 10 * prefix.shape[1], 10)
    rec_delivery = recognised[p]; prefix = prefix.copy()
    for s in np.unique(sessions):
        rows = np.flatnonzero(sessions == s)
        first = np.flatnonzero(rec_delivery[rows])
        stop = int(first[0]) if len(first) else len(rows)
        prefix[np.ix_(rows, np.flatnonzero((grid >= stop) | (grid > CAP)))] = np.nan
    validate_prefix(prefix, pos, sessions, grid, rec_delivery)
    check = compare_package_stages(b, features, pc, em_out, out, prefix,
                                   pos, sessions, rec_delivery, grid)
    check["causality"] = check_causality(S, chain, scope, p, base, sessions, pos, out, calls[-1][4])
    o = np.argsort(p)
    return dict(index=idx, order=np.array(order), base=np.array(base), emitted=emitted, prefix=prefix[o], pos=pos[o], grid=grid,
                recognised=recognised, trace_sha256=np.array(hashlib.sha256(trace_path.read_bytes()).hexdigest()),
                package_sha256=np.array(PACKAGE_SHA), stage_check=np.array(str(check)))


def main():
    assert len(sys.argv) == 3, "supply one delivery order and base"
    order, base = sys.argv[1:]; out = ROOT / f"runs/s281_seen_relabel/{order}_{base}.npz"
    assert not out.exists(), f"refusing to overwrite {out}"
    result = produce(order, base)
    out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, **result)
    print(f"saved {out}; package stage parity {result['stage_check']}", flush=True)


if __name__ == "__main__":
    main()
