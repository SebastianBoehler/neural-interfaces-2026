"""Batch eligible frozen base members in a new archive, preserving order and CPU arithmetic."""
from pathlib import Path
import sys

from patch_terminal import HERE, once
from build_cb_speed import build

NETBANK = "class NetBank:\n"
GROUPS = "        self.groups, self.dev = [g for g in groups.values() if g[1]], dev\n"
CALL = """            inputs, cache = {}, {}
            for i, net in members:
                if net.dtype not in inputs:
                    inputs[net.dtype] = H.to(dtype=net.dtype).to(self.dev)
                ids.append(i)
                probs.append(net.proba(inputs[net.dtype], cache))
"""
BATCH_CALL = """            inputs, cache, outputs = {}, {}, {}
            for i, net in members:
                if net.dtype not in inputs:
                    inputs[net.dtype] = H.to(dtype=net.dtype).to(self.dev)
                ids.append(i)
                batch = self.frozen_batches.get(i)
                if batch is None:
                    probs.append(net.proba(inputs[net.dtype], cache))
                    continue
                if i not in outputs:
                    group, batched = batch
                    outputs.update(zip(group, batched.proba(inputs[net.dtype], cache), strict=True))
                probs.append(outputs[i])
"""


def patch(source):
    batch = (HERE / "base_batches.py.inc").read_text().rstrip()
    text = once(source, NETBANK, batch + "\n\n\n" + NETBANK)
    text = once(text, GROUPS, GROUPS + """        self.frozen_batches = {}
        if torch.device(dev).type == "cuda":
            self.frozen_batches = build_frozen_batches(self.groups)
            if not self.frozen_batches:
                raise ValueError("no eligible frozen member batch")
""")
    text = once(text, CALL, BATCH_CALL)
    compile(text, "submission.py", "exec")
    return text


if __name__ == "__main__":
    parent, digest, target = sys.argv[1:]
    build(Path(parent), Path(target), expected_sha=digest, patch_source=patch)
