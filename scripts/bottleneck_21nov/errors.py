"""Decompose the official half-new/half-returning balanced-accuracy error."""
from collections import defaultdict

import numpy as np

from tapes import PEOPLE, POSITIONS, ROSTERS, joint, mass, paired, score, subsets


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |",
                      "| " + " | ".join("---" for _ in head) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def fmt_ci(values):
    point, low, high = values
    return f"{point:+.2f} [{low:+.2f}, {high:+.2f}]"


def cell_stats(cases):
    counts = {}
    values = defaultdict(lambda: defaultdict(list))
    for case in cases:
        for interface in np.unique(case.interface):
            cell = (case.cohort, case.person, case.visit, interface)
            mask = case.interface == interface
            counts[cell] = tuple(np.bincount(case.y[mask], minlength=3).tolist())
            for roster in ROSTERS:
                values[cell][roster].append(100 * case.weights[mask] @ (case.scores[roster][mask] == case.y[mask]))
    return {key: (counts[key], {roster: np.mean(x) for roster, x in v.items()}) for key, v in values.items()}


def class_confusion(cases, roster, cohort, interface):
    cc = np.zeros((3, 3))
    for case, mask in subsets(cases, cohort=cohort, interface=interface):
        np.add.at(cc, (case.y[mask], case.scores[roster][mask]), case.weights[mask])
    return cc


def lost(cases, roster, cohort, **filters):
    fn = lambda case: case.scores[roster] != case.y
    return 50 * mass(cases, fn, cohort, **filters)[0] / mass(cases, fn, cohort)[1]


def error_summary(cases, orders, interfaces):
    models = [lambda case, name=name: case.scores[name] for name in ROSTERS]
    sizes = cell_stats(cases)
    n_cells = {c: sum(k[0] == c for k in sizes) for c in PEOPLE}
    labels = ["Held-new", "Returning"]
    rows = []
    for roster, get in zip(ROSTERS, models, strict=True):
        vals = [score(cases, get, c) for c in PEOPLE]
        rows.append((roster, *(f"{v:.2f}" for v in vals), f"{np.mean(vals):.2f}",
                     *(f"{(100 - v) / 2:.2f}" for v in vals)))
    output = [table(["Stack", *labels, "50/50 BA", "New lost pp", "Ret. lost pp"], rows)]
    delta = paired(cases, models[1], models[0], orders)
    output.append("Paired ba_six gain in pp [80% interval], held-new / returning / 50/50: "
                  + " · ".join(fmt_ci(delta[c]) for c in ("new", "ret", "J")) + ".")
    if abs(delta["J"][0] - 0.3566) > 0.0001:
        raise ValueError(f"failed to reproduce reference 50/50 delta: {delta}")
    if not np.allclose(delta["J"][1:], (.17, .54), rtol=0, atol=.008):
        raise ValueError(f"failed to reproduce reference 80% interval: {delta}")
    rows = []
    for person in PEOPLE["new"]:
        vals = []
        for cohort in PEOPLE:
            current = [v[1] for k, v in sizes.items() if k[:2] == (cohort, person)]
            vals += [f"{np.mean([x[r] for x in current]):.2f}" if current else "—" for r in ROSTERS]
        rows.append((person, *vals))
    output.extend(["### People", table(["Person", "New nine25", "New ba_six", "Ret. nine25", "Ret. ba_six"], rows)])
    rows = []
    for cohort, person, visit in sorted({k[:3] for k in sizes}):
        summary = []
        for interface in interfaces:
            n, v = sizes[cohort, person, visit, interface]
            summary.append(f"{sum(n)} · {v[ROSTERS[0]]:.1f}/{v[ROSTERS[1]]:.1f} · "
                           f"{(100 - v[ROSTERS[0]]) / (2*n_cells[cohort]):.2f}/"
                           f"{(100 - v[ROSTERS[1]]) / (2*n_cells[cohort]):.2f}")
        rows.append((cohort, person, visit, *summary))
    output.extend(["### Sessions and cells",
                   "Each cell shows window count · nine25/ba_six BA% · nine25/ba_six lost 50/50 pp.",
                   table(["Cohort", "Person", "Visit", *interfaces], rows)])
    top = sorted(sizes, key=lambda k: (100 - sizes[k][1]["ba_six"]) / n_cells[k[0]], reverse=True)[:5]
    ranked = [f"{p}/{s} {i} ({c}, {(100 - sizes[(c,p,s,i)][1]['ba_six']) / (2*n_cells[c]):.2f} pp)"
              for c, p, s, i in top]
    output.append("Largest ba_six cell losses in rank order: " + ", ".join(ranked) + ".")
    rows = []
    for cohort in PEOPLE:
        for label, dims in [*((interface, {"interface": interface}) for interface in interfaces),
                            *((name, {"position": i}) for i, name in enumerate(POSITIONS))]:
            result = [f"{score(cases, get, cohort, **dims):.2f} / {lost(cases, r, cohort, **dims):.2f}"
                      for r, get in zip(ROSTERS, models, strict=True)]
            rows.append((cohort, label, *result))
    output.extend(["### Interface and position", "Each model shows BA% / lost 50/50 pp. Position bins reuse "
                   "full-cell class weights and renormalize within the bin for accuracy. Losses add across bins.",
                   table(["Cohort", "Stratum", *ROSTERS], rows)])
    rows = []
    for cohort in PEOPLE:
        for iface in interfaces:
            for roster in ROSTERS:
                cc = class_confusion(cases, roster, cohort, iface)
                recalls = cc / cc.sum(1, keepdims=True) * 100
                pairs = {(i, j): cc[i, j] + cc[j, i] for i, j in ((0, 1), (0, 2), (1, 2))}
                pair = max(pairs, key=pairs.get)
                share = 100 * pairs[pair] / (cc.sum() - cc.trace())
                rows.append((cohort, iface, roster, *[" / ".join(f"{v:.1f}" for v in row) for row in recalls],
                             f"{pair[0]}↔{pair[1]} · {share:.1f}%"))
    output.extend(["### Confusion", "Rows show true-class recall distributions. Columns within each triple are "
                   "0 motor imagery, 1 mental subtraction, 2 word generation. The final column gives the largest "
                   "bidirectional pair and its share of off-diagonal mass.",
                   table(["Cohort", "Interface", "Stack", "True 0", "True 1", "True 2", "Top pair"], rows)])
    rows = []
    for cohort in PEOPLE:
        for roster in ROSTERS:
            class_loss = []
            for c in range(3):
                p, w = mass(cases, lambda case: (case.y == c) & (case.scores[roster] != c), cohort)
                class_loss.append(f"{50 * p / w:.2f}")
            rows.append((cohort, roster, *class_loss))
    output.append(table(["Cohort", "Stack", "0 lost pp", "1 lost pp", "2 lost pp"], rows))
    groups = defaultdict(list)
    for key, (counts, vals) in sizes.items():
        groups[sum(counts)].append((key[0], vals))
    rows = []
    for size, grp in sorted(groups.items()):
        mass_by_r = [sum((100 - v[r]) / (2 * n_cells[c]) for c, v in grp) for r in ROSTERS]
        rows.append((size, sum(c == "new" for c, _ in grp), sum(c == "ret" for c, _ in grp),
                     *(f"{v:.2f}" for v in mass_by_r)))
    output.extend(["### Cell size", table(["Windows", "New cells", "Ret. cells", "Nine25 lost pp", "ba_six lost pp"], rows)])
    for roster in ROSTERS:
        pred = lambda case: case.scores[roster]
        cell_loss = sum((100 - vals[roster]) / (2 * n_cells[key[0]]) for key, (_, vals) in sizes.items())
        np.testing.assert_allclose(cell_loss, 100 - joint(cases, pred), rtol=0, atol=1e-8)
    return "\n\n".join(output)
