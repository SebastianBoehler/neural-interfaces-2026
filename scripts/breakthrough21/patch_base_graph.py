"""Capture the base NetBank's CUDA calls without changing its CPU or training paths."""
from pathlib import Path
import sys

from patch_terminal import HERE, once
from build_cb_speed import build

NETBANK = "class NetBank:\n"
GROUPS = "        self.groups, self.dev = [g for g in groups.values() if g[1]], dev\n"
RESET = """    def reset_state(self):
        for stream, members in self.groups:
"""
CALL = """    def __call__(self, x):
        ids, probs, self.last_inputs = [], [], {}
"""


def patch(source):
    graph = (HERE / "base_graph.py.inc").read_text().rstrip()
    text = once(source, NETBANK, graph + "\n\n\n" + NETBANK)
    text = once(text, GROUPS, GROUPS + """        self.use_cuda_graph = torch.device(dev).type == "cuda"
        self.base_graph = BaseMemberGraph() if self.use_cuda_graph else None
""")
    text = once(text, RESET, """    def reset_state(self):
        self.base_graph = BaseMemberGraph() if self.use_cuda_graph else None
        for stream, members in self.groups:
""")
    text = once(text, CALL, """    def __call__(self, x):
        if self.base_graph is not None:
            return self.base_graph.run(self, x)
        ids, probs, self.last_inputs = [], [], {}
""")
    compile(text, "submission.py", "exec")
    return text


if __name__ == "__main__":
    parent, digest, target = sys.argv[1:]
    build(Path(parent), Path(target), expected_sha=digest, patch_source=patch)
