"""Build the two hard-only Team B ZIPs from the signed deployment archives."""

import argparse
import ast
from copy import copy
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import torch

from build_cb_speed import digest, verify_parent
from build_team_b_speed import once
from patch_dead import NEW
from patch_selftest import ATOL, WINDOW, windows

BASE = "final_stack_hard_safe_st"
BASE_SHA = "239c86bd18d0a7e9d0216dda1fb95d9b918bf6e933c0383322446aed37a83c1e"
DONOR = "final_stack_hardcb_safe_team_b_nine25_cbgraph_tbgraph"
DONOR_SHA = "911a31af329c3f38f75ae8a0f1331e18bc921e6c36b002538839ba04a2384c4e"
DOSES = {"nine15": 0.15, "nine25": 0.25}
SOLVER = "class Solver(CompetSolver):"

SELF_TEST = '''
    @torch.no_grad()
    def _check_team_b(self, dev):
        # Check the signed encoder on the real device before any session can fall back.
{window}        expected = np.array({expected})
        try:
            actual = np.stack([self.team_b.posterior(torch.from_numpy(x)).cpu().numpy() for x in _xs])
            if torch.device(dev).type == "cuda":
                features = self.team_b._embed(torch.stack(self.team_b.windows * 32))
                if len(features) != 9 or not all(torch.isfinite(f).all().item() for f in features):
                    raise RuntimeError("invalid Team B features for batch 64")
                for size in (1, 64):
                    graph = self.team_b.feature_graphs[size]
                    if graph.graph is None or graph.failed or graph.replays != 1:
                        raise RuntimeError(f"Team B feature graph failed for batch {{size}}")
        except Exception as exc:
            raise RuntimeError(f"load-time self-test: Team B raised {{exc!r}}") from exc
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol={atol}):
            raise RuntimeError(f"load-time self-test: Team B posterior {{actual}} != reference {{expected}}")
        self.team_b.reset()
'''


def name(arm):
    return f"{BASE}_team_b_{arm}_tbgraph.zip"


def read_donor(path):
    verify_parent(path, DONOR_SHA)
    with ZipFile(path) as donor:
        source = donor.read("submission.py").decode()
        blob = donor.read("team_b.pt")
    lines = source.splitlines(keepends=True)
    nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    forward = next(n for n in nodes["CBMember"].body if isinstance(n, ast.ClassDef) and n.name == "ForwardGraph")
    graph = dedent("".join(lines[forward.lineno - 1:forward.end_lineno]))
    graph = once(graph, "class ForwardGraph:", "class TeamBFeatureGraph:")
    graph = once(graph, "CBraMod CUDA graph capture failed", "Team B CUDA graph capture failed")
    start = source.index('_TEAM_B = {"_FeatureGraph": CBMember.ForwardGraph}')
    end = source.index("class TeamBlendStack(", start)
    namespace = once(source[start:end], "CBMember.ForwardGraph", "TeamBFeatureGraph")
    team = nodes["TeamBlendStack"]
    stack = "".join(lines[team.lineno - 1:team.end_lineno])
    return graph + "\n\n" + namespace, stack, blob


def reference(namespace, blob, members=3):
    torch.set_num_threads(1)
    ns = {"torch": torch}
    exec(compile(namespace, "<signed Team B encoder>", "exec"), ns)
    bank = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    encoder = ns["_TEAM_B"]["Encoder"](bank, "cpu", members)
    posteriors = np.stack([encoder.posterior(torch.from_numpy(x)).numpy() for x in windows()])
    if posteriors.shape != (2, 3) or not np.isfinite(posteriors).all() or (posteriors < 0).any():
        raise ValueError("invalid Team B self-test reference")
    np.testing.assert_allclose(posteriors.sum(1), 1, rtol=0, atol=1e-6)
    return posteriors.tolist()


def patched(source, namespace, stack, expected, dose):
    if source.count(NEW) != 1 or source.count("# load-time self-test (patch_selftest.py)") != 1:
        raise ValueError("expected the signed hard self-test and dead-session guard")
    stack = once(stack, "class TeamBlendStack(StackEnsemble):", "class TeamBlendStack(HardWaveEnsemble):")
    stack = once(stack, "    TEAM_B_MIX = 0.25\n", f"    TEAM_B_MIX = {dose}\n")
    anchor = '            self.team_b = _TEAM_B["Encoder"](pack, dev, self.TEAM_B_MEMBERS)\n'
    stack = once(stack, anchor, anchor + "            self._check_team_b(dev)\n")
    test = SELF_TEST.format(window=WINDOW, expected=repr(expected), atol=ATOL)
    source = once(source, SOLVER, namespace + stack + test + "\n\n" + SOLVER)
    source = once(source, "model = HardWaveEnsemble(z,", "model = TeamBlendStack(z,")
    compile(source, "submission.py", "exec")
    return source.encode()


def build(zips, out):
    parent = zips / f"{BASE}.zip"
    verify_parent(parent, BASE_SHA)
    namespace, stack, blob = read_donor(zips / f"{DONOR}.zip")
    expected = reference(namespace, blob)
    targets = [out / name(arm) for arm in DOSES]
    for target in targets:
        if target.exists():
            raise FileExistsError(target)
    with TemporaryDirectory(dir=out) as scratch, ZipFile(parent) as before:
        submission = before.read("submission.py").decode()
        entries = before.namelist() + ["team_b.pt"]
        for (arm, dose), target in zip(DOSES.items(), targets, strict=True):
            source = patched(submission, namespace, stack, expected, dose)
            staged = Path(scratch) / name(arm)
            with ZipFile(staged, "x") as after:
                for entry in before.infolist():
                    data = source if entry.filename == "submission.py" else before.read(entry)
                    after.writestr(copy(entry), data)
                after.writestr(ZipInfo("team_b.pt"), blob, compress_type=ZIP_DEFLATED)
            with ZipFile(staged) as after:
                if after.namelist() != entries or after.testzip() is not None:
                    raise ValueError("hard Team B ZIP inventory or CRC differs")
                for entry in entries:
                    ref = {"submission.py": source, "team_b.pt": blob}
                    if after.read(entry) != (ref[entry] if entry in ref else before.read(entry)):
                        raise ValueError(f"hard Team B ZIP changed {entry}")
        for target in targets:
            os.link(Path(scratch) / target.name, target)
            print(digest(target), target.stat().st_size, target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zips", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    build(args.zips, args.out)
