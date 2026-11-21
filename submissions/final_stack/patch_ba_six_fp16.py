"""Use Turing tensor cores for the signed MANAS encoder; preserve the BF16 and CPU paths."""
import argparse
import ast
from pathlib import Path

from build_cb_speed import build
from patch_team_b_bank import once

PARENT_SHA = "fc2acf89bc194e7e576dd9fa9b6a6858207d06fe98396a1e4e1f785c04e9a5aa"
NIV_SHA = "3c4796d5143d307fdfad0004557b3ceffbd67f2ff18d76068b4e4083d8eaaf29"
FORWARD = '''        bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
'''
TURING = '''        cuda = self.device.type == "cuda"
        bf16 = cuda and torch.cuda.is_bf16_supported()
        fp16 = cuda and not bf16 and torch.cuda.get_device_capability(self.device) == (7, 5)
        dtype = torch.float16 if fp16 else torch.bfloat16
        with torch.autocast(self.device.type, dtype=dtype, enabled=bf16 or fp16):
'''


def patch(source):
    marker = "_TEAM_A = {}\nexec(compile("
    start = source.index(marker) + len(marker)
    stop = source.index(', "<MANAS A1>", "exec"), _TEAM_A)', start)
    old = ast.literal_eval(source[start:stop])
    new = once(old, FORWARD, TURING)
    compile(new, "<MANAS A1>", "exec")
    return source[:start] + repr(new) + source[stop:]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path)
    p.add_argument("target", type=Path)
    p.add_argument("kind", choices=("production", "niv"))
    a = p.parse_args()
    build(a.parent, a.target, expected_sha=PARENT_SHA if a.kind == "production" else NIV_SHA, patch_source=patch)
