"""Log EEG identity decisions without opening trial labels."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from data import plans
from stream import load_solver


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--orders", default="71000,71001")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    with np.load(a.data, allow_pickle=False) as z:
        who = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]])
        visit, names = z["session"], z["ch_names"]
        x = torch.from_numpy(z["X"])
    stream = {"stream": np.char.add(np.char.add(who, "|"), visit)}
    report = []
    for row in (a.run / "jobs.tsv").read_text().splitlines():
        role, held, people, _, sessions = row.split("\t")
        pkg = a.run / f"{role}_{held}"
        names_by_index = sorted(people.split(","))
        template = load_solver(pkg, names, "cuda")
        wanted = {f"{held}|{s}" for s in (np.unique(visit[who == held]) if sessions == "all" else sessions.split(","))}
        with np.load(a.run / f"pred_{role}_{held}.npz", allow_pickle=False) as expected:
            for order in map(int, a.orders.split(",")):
                for session, rows in plans(stream, order, sorted(wanted)).items():
                    model = copy.deepcopy(template)
                    recognition, predictions = None, []
                    for age, i in enumerate(rows, 1):
                        predictions.append(int(model.predict(x[i:i + 1].cuda())[0]))
                        if model.person is not None and recognition is None:
                            recognition = dict(person=names_by_index[model.person], age=age)
                    visit_id = session.split("|")[1]
                    np.testing.assert_array_equal(rows, expected[f"rows_{order}_{visit_id}"])
                    np.testing.assert_array_equal(predictions, expected[f"labels_{order}_{visit_id}"])
                    record = dict(role=role, session=session, order=order, recognition=recognition)
                    report.append(record)
                    print(record, flush=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
