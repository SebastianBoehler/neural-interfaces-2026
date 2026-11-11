"""Person-equal scoring helpers for the TSEM confirmation (cell-mean BA per person, pe_metric definition)."""
import numpy as np
import paths
B = dict(np.load(paths.DATA + "/bases.npz", allow_pickle=True))
PROTOS = ("all_loso", "seen_session")
PEOPLE = [f"P{i:02d}" for i in range(1, 11)]; SEEN = ["P04", "P10"]
G1 = ("P01", "P02", "P05", "P07", "P09"); G2 = ("P03", "P04", "P06", "P08", "P10")


def cell_ba(y, pred):
    return 100 * np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)])


def person(P, proto):
    y, cell, per = B[f"y|{proto}"], B[f"cell|{proto}"], B[f"person|{proto}"]; pred = P.argmax(1); acc = {}
    for c in np.unique(cell):
        m = cell == c; acc.setdefault(per[m][0], []).append(cell_ba(y[m], pred[m]))
    ref = PEOPLE if proto == "all_loso" else SEEN
    return np.array([np.mean(acc[p]) for p in ref])


def session(P, proto):
    """-> {label: BA} per session (label e.g. P04s2; seen_session sessions numbered from 2)."""
    y, cell, per, ss = B[f"y|{proto}"], B[f"cell|{proto}"], B[f"person|{proto}"], B[f"sess|{proto}"]
    pred = P.argmax(1); out = {}; cnt = {}
    for s in np.unique(ss):
        m = ss == s; p = per[m][0]; cnt[p] = cnt.get(p, 0) + 1
        out[f"{p}s{cnt[p] + (proto == 'seen_session')}"] = np.mean([cell_ba(y[m & (cell == c)], pred[m & (cell == c)]) for c in np.unique(cell[m])])
    return out


def J(dn, ds, people=None):
    people = PEOPLE if people is None else people
    a = np.mean([dn[PEOPLE.index(p)] for p in people]); b = [ds[SEEN.index(p)] for p in people if p in SEEN]
    return 0.5 * a + 0.5 * np.mean(b) if b else a


def compare(Pa, Pb):
    """Deltas of arm Pa vs base Pb ({proto: P}) -> dict."""
    dn = person(Pa["all_loso"], "all_loso") - person(Pb["all_loso"], "all_loso")
    ds = person(Pa["seen_session"], "seen_session") - person(Pb["seen_session"], "seen_session")
    sa = {p: session(Pa[p], p) for p in PROTOS}; sb = {p: session(Pb[p], p) for p in PROTOS}
    sess = {("new " if p == "all_loso" else "seen ") + k: sa[p][k] - sb[p][k] for p in PROTOS for k in sa[p]}
    return dict(J=J(dn, ds), new=dn.mean(), seen=ds.mean(), JG1=J(dn, ds, G1), JG2=J(dn, ds, G2), dn=dn, ds=ds,
                npos=int((dn >= 0).sum()), sess=sess)


def line(tag, r):
    return (f"{tag:34s} dJ {r['J']:+.2f} new {r['new']:+.2f} seen {r['seen']:+.2f} | JG1 {r['JG1']:+.2f} JG2 {r['JG2']:+.2f} "
            f"| new>=0 {r['npos']}/10 | new " + " ".join(f"{p[1:]}:{v:+.1f}" for p, v in zip(PEOPLE, r["dn"]))
            + " | seen " + " ".join(f"{p[1:]}:{v:+.1f}" for p, v in zip(SEEN, r["ds"])))


def absline(tag, P):
    n, s = person(P["all_loso"], "all_loso"), person(P["seen_session"], "seen_session")
    return f"{tag:34s} abs J {0.5*n.mean()+0.5*s.mean():.2f} new {n.mean():.2f} seen {s.mean():.2f}"
