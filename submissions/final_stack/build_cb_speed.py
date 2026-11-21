"""Derive the faster hardcb archive from the pinned plain self-test ZIP."""

import argparse
from copy import copy
from hashlib import file_digest
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
NAME = "final_stack_hardcb_safe_st"
PARENT_SHA = "47faa27b25f6fc1fcbcc1c74f2be4b8b2aeb2a1e02c6c957d342eb6c91b9c2ec"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify_parent(path):
    if digest(path) != PARENT_SHA:
        raise ValueError(f"{path}: unexpected parent SHA-256")


def patch(source):
    original = (HERE / "cb_member.py.inc").read_text().rstrip()
    methods = (HERE / "cb_graph.py.inc").read_text().rstrip()
    if source.count(original) != 1:
        raise ValueError("expected exactly one unmodified CBMember")
    start, rest = original.split("        aligned = None\n")
    _, end = rest.split("        if not np.isfinite(q).all()")
    start = start.replace("        self.n = 0\n", "        self.n = 0\n        self._forward_graph = self.ForwardGraph()\n")
    updated = start + "        q = self._posterior(x)\n        if not np.isfinite(q).all()" + end
    result = source.replace(original, updated + "\n" + methods)
    compile(result, "submission.py", "exec")
    return result


def verify_pair(parent, candidate):
    verify_parent(parent)
    with ZipFile(parent) as before, ZipFile(candidate) as after:
        if before.namelist() != after.namelist() or after.testzip() is not None:
            raise ValueError(f"{candidate}: changed ZIP inventory or bad CRC")
        changed = [name for name in before.namelist() if before.read(name) != after.read(name)]
        if changed != ["submission.py"] or patch(before.read("submission.py").decode()) != after.read("submission.py").decode():
            raise ValueError(f"{candidate}: unexpected source or artifact change")


def build(parent, candidate):
    verify_parent(parent)
    with ZipFile(parent) as before, ZipFile(candidate, "x") as after:
        for entry in before.infolist():
            payload = before.read(entry)
            if entry.filename == "submission.py":
                payload = patch(payload.decode()).encode()
            after.writestr(copy(entry), payload)
    verify_pair(parent, candidate)
    print(digest(candidate), candidate, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parents", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    build(args.parents / f"{NAME}.zip", args.out / f"{NAME}_cbgraph.zip")


if __name__ == "__main__":
    main()
