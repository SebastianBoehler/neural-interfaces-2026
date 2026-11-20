"""Write the official single-file Solver from the frozen Team C and M1 sources."""
import json
from pathlib import Path

from base import C

HERE = Path(__file__).resolve().parent
ARMS = ("context", "blank", "a0", "a1", "a2h")

SOLVER = """
class Solver(CompetSolver):
    name = "m1"

    def load_model(self, meta):
        torch.backends.cudnn.benchmark = False
        folder = Path(meta["submission_dir"])
        dev = meta.get("device", "cpu")
        arm = json.loads((folder / "config.json").read_text())["arm"]
        cfg = {
            "context": dict(bn="source", steps=0),
            "blank": dict(bn="source", steps=0),
            "a0": dict(bn="source", steps=0),
            "a1": dict(bn="prefix", steps=0),
            "a2h": dict(bn="prefix", steps=8, lr=3e-4),
        }[arm]
        base = load_package(folder, dev, cfg)
        if arm in ("context", "blank"):
            context = Context(len(base.nets)).to(dev)
            context.load_state_dict(torch.load(folder / "context.pt", map_location=dev, weights_only=True))
            return ContextStream(base, context.eval(), arm == "context")
        return base
"""


def write_packages(root, weights, context):
    c1 = Path(C.__file__).read_text()
    m1 = (HERE / "context.py").read_text()
    body = f"{c1}\n\n{m1}\n\nimport json\n{SOLVER}"
    for arm in ARMS:
        path = root / arm
        path.mkdir(parents=True, exist_ok=True)
        (path / "submission.py").write_text(body)
        for filename, target in [("weights.pt", weights), ("context.pt", context)]:
            link = path / filename
            link.unlink(missing_ok=True)
            link.symlink_to(target.resolve(strict=True))
        (path / "config.json").write_text(json.dumps({"arm": arm}))
