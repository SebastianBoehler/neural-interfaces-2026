"""Derive a nine25 archive that reuses CBraMod and optional Team B CUDA graphs."""

import argparse
import ast
from functools import partial
from pathlib import Path
from zipfile import ZipFile

from build_cb_speed import HERE, NAME, PARENT_SHA, build, patch, verify_pair, verify_parent

TEAM_B = "final_stack_hardcb_safe_team_b_nine25"
TEAM_B_SHA = "af40e37507996302e94f2d69df6beec20405cb6932965a5766352d3cb44b9c79"


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"unexpected Team B source anchor {old[:80]!r}")
    return text.replace(old, new, 1)


def constructor_test(zips):
    """Reuse the signed fixed-window self-test for the identical hard / CB bank."""
    path = zips / f"{NAME}.zip"
    verify_parent(path, PARENT_SHA)
    with ZipFile(path) as plain, ZipFile(zips / f"{TEAM_B}.zip") as team:
        actor_names = [n for n in plain.namelist() if n.startswith(("hard_s", "cb_s"))]
        if len(actor_names) != 5 or any(plain.read(n) != team.read(n) for n in actor_names):
            raise ValueError("nine25's hard or CBraMod weights differ from the self-test reference")
        src = plain.read("submission.py").decode()
    start = src.index("        # load-time self-test (patch_selftest.py)")
    return src[start:src.index("\n    def reset_state", start)]


def graph_encoder(source):
    marker = "_TEAM_B = {}\nexec(compile("
    start = source.index(marker) + len(marker)
    stop = source.index(", '<Team B encoder>', 'exec'), _TEAM_B)", start)
    encoder = ast.literal_eval(source[start:stop])
    old = (HERE / "team_b/encoder.py.inc").read_text()
    reset = "        self.windows, self.cov, self.embeddings = [], None, None\n"
    updated = once(old, reset, reset + '        self.feature_graphs = {n: _FeatureGraph() for n in (1, 64)} if torch.device(self.device).type == "cuda" else {}\n')
    updated = once(updated,
                   "            self.embeddings = [\n"
                   "                torch.cat([net.feats(xs[i:i + 64]) for i in range(0, n, 64)]) for net in self.nets\n"
                   "            ]",
                   "            self.embeddings = self._embed_prefix(xs)")
    updated = once(updated,
                   "                torch.cat([emb, net.feats(xa)]) for net, emb in zip(self.nets, self.embeddings, strict=True)",
                   "                torch.cat([emb, new]) for emb, new in zip(self.embeddings, self._embed(xa), strict=True)")
    encoder = once(encoder, old, updated + (HERE / "team_b/feature_graph.py.inc").read_text())
    compile(encoder, "<Team B encoder>", "exec")
    source = source[:start] + repr(encoder) + source[stop:]
    return once(source, "_TEAM_B = {}", '_TEAM_B = {"_FeatureGraph": CBMember.ForwardGraph}')


def gpu_graph_input(source):
    source = once(source,
                  "                host = torch.empty_like(aligned, pin_memory=True)\n"
                  "                host.copy_(aligned)\n"
                  "                static = host.to(dev)",
                  "                host = None\n"
                  "                if not aligned.is_cuda:\n"
                  "                    host = torch.empty_like(aligned, pin_memory=True)\n"
                  "                    host.copy_(aligned)\n"
                  "                static = (aligned if host is None else host).to(dev, copy=True)")
    return once(source,
                "            self.host.copy_(aligned)\n"
                "            self.static.copy_(self.host, non_blocking=True)",
                "            if self.host is not None:\n"
                "                self.host.copy_(aligned)\n"
                "            self.static.copy_(aligned if self.host is None else self.host, non_blocking=True)")


def patch_team_b(source, *, self_test, team_graph):
    source = patch(source)
    anchor = '            raise ValueError("the stack package has no member")\n'
    source = once(source, anchor, anchor + self_test)
    if team_graph:
        source = graph_encoder(gpu_graph_input(source))
    compile(source, "submission.py", "exec")
    return source


def fixture(zips, team_graph):
    name = f"{TEAM_B}_cbgraph" + ("_tbgraph" if team_graph else "") + ".zip"
    return zips / f"{TEAM_B}.zip", zips / name


def verify(zips, team_graph):
    paths = fixture(zips, team_graph)
    src = partial(patch_team_b, self_test=constructor_test(zips), team_graph=team_graph)
    verify_pair(*paths, expected_sha=TEAM_B_SHA, patch_source=src)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parents", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--team-graph", action="store_true")
    args = parser.parse_args()
    parent, candidate = fixture(args.parents, args.team_graph)
    src = partial(patch_team_b, self_test=constructor_test(args.parents), team_graph=args.team_graph)
    build(parent, args.out / candidate.name, expected_sha=TEAM_B_SHA, patch_source=src)


if __name__ == "__main__":
    main()
