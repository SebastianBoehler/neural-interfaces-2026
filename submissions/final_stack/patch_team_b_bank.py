"""Derive a signed Team B bank variant in a new ZIP from the nine25 graph parent."""

import argparse
import ast
from copy import copy
import hashlib
import io
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from build_ba_terminal import PARENT_SHA, PRODUCTION_SHA, eager_reference, partner_source, patch as patch_manas
from build_hard_team_b import once, reference, verify_parent

FAMILIES = ("shallow", "slow", "fbc")
COUNTS = {"nine25": (3, 3, 3), "swap25": (3, 3, 3), "fbc12": (3, 3, 6), "ba_six": (2, 2, 2),
          "triseed25": (3, 3, 3), "ba_seven": (2, 2, 3), "ba_eight": (3, 2, 3)}
MANAS_ARMS = ("ba_six", "ba_seven", "ba_eight")


def bank_count(arm):
    return 3 if arm == "triseed25" else 2 if arm in ("swap25", "fbc12") else 1


def roster(counts):
    return tuple(kind for kind, count in zip(FAMILIES, counts, strict=True) for _ in range(count))


def load_bank(blob):
    pack = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    if tuple(kind for kind, _ in pack["nets"]) != roster(COUNTS["nine25"]):
        raise ValueError("expected three members per family")
    if pack["steps"] != 1200 or pack["lam"] != 0.5 or any(pack.get(k) for k in ("experts", "ext", "cfg")):
        raise ValueError("expected the signed 1,200-step R1 recipe")
    return pack


def bank_for(arm, original, extra=None, third=None):
    pack = load_bank(original)
    if arm == "nine25":
        return original
    extra_packs = [load_bank(b) for b in (extra, third)[:bank_count(arm) - 1]]
    for donor in extra_packs:
        if (pack["people"], pack["first_only"]) != (donor["people"], donor["first_only"]):
            raise ValueError("the two Team B banks use different fold rosters")
    nets = []
    for family, i in enumerate((0, 3, 6)):
        chunk = pack["nets"][i:i + 3]
        if arm == "swap25":
            chunk = chunk[:2] + extra_packs[0]["nets"][i:i + 1]
        elif arm == "fbc12" and i == 6:
            chunk += extra_packs[0]["nets"][i:i + 3]
        elif arm == "triseed25":
            chunk = chunk[:1] + [bank["nets"][i] for bank in extra_packs]
        elif arm in MANAS_ARMS:
            chunk = chunk[:COUNTS[arm][family]]
        for j, (kind, incoming) in enumerate(chunk):
            if any(all(torch.equal(v, incoming[k]) for k, v in state.items()) for _, state in chunk[:j]):
                raise ValueError(f"duplicate selected member for {kind}")
        nets.extend(chunk)
    if tuple(kind for kind, _ in nets) != roster(COUNTS[arm]):
        raise ValueError(f"unexpected selected Team B roster for {arm}")
    pack["nets"] = nets
    out = io.BytesIO()
    torch.save(pack, out)
    return out.getvalue()


def patch_bank(source, arm, blob):
    counts = COUNTS[arm]
    members = counts[0]
    if len(set(counts)) > 1:
        marker = 'exec(compile('
        start = source.index(marker, source.index("_TEAM_B = ")) + len(marker)
        stop = source.index(", '<Team B encoder>', 'exec'), _TEAM_B)", start)
        encoder = ast.literal_eval(source[start:stop])
        encoder = once(encoder, '("shallow",) * members + ("slow",) * members + ("fbc",) * members', repr(roster(counts)))
        source = source[:start] + repr(encoder) + source[stop:]
    namespace = source[source.index("class TeamBFeatureGraph:"):source.index("class TeamBlendStack(")]
    expected = reference(namespace, blob, members)
    source = once(source, "    TEAM_B_MEMBERS = 3\n", f"    TEAM_B_MEMBERS = {members}\n")
    signature = next(line for line in source.splitlines() if line.startswith("    TEAM_B_SHA = "))
    source = once(source, signature, f"    TEAM_B_SHA = {hashlib.sha256(blob).hexdigest()!r}")
    team_test = source.split("    def _check_team_b(self, dev):", 1)[1]
    scores = next(line for line in team_test.splitlines() if line.startswith("        expected = np.array("))
    source = once(source, scores, f"        expected = np.array({expected!r})")
    source = once(source, "if len(features) != 9 or", f"if len(features) != {sum(counts)} or")
    compile(source, "submission.py", "exec")
    return source


def with_manas(source, donor, weights_sha, license_text):
    original = (donor / "submission.py").read_text()
    expected = eager_reference(donor, original)
    return patch_manas(source, partner_source(original), license_text, weights_sha, expected, True).decode()


def archive(parent, out, arm, extra=None, manas_zip=None, third=None):
    verify_parent(parent, PARENT_SHA)
    if (arm in MANAS_ARMS) != (manas_zip is not None) or bank_count(arm) != 1 + (extra is not None) + (third is not None):
        raise ValueError(f"incomplete donor set for {arm}")
    with ZipFile(parent) as before, TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / out.name
        blob = bank_for(arm, before.read("team_b.pt"), extra, third)
        source = patch_bank(before.read("submission.py").decode(), arm, blob)
        names = before.namelist()
        if manas_zip:
            verify_parent(manas_zip, PRODUCTION_SHA)
            donor = Path(tmp) / "manas"
            with ZipFile(manas_zip) as package:
                package.extractall(donor, members=("submission.py", "weights.pt", "LICENSE-MANAS.txt"))
            weights = donor / "weights.pt"
            with weights.open("rb") as stream:
                weights_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            source = with_manas(source, donor, weights_sha, (donor / "LICENSE-MANAS.txt").read_text())
            names.append("team_a.pt")
        payloads = {"submission.py": source.encode(), "team_b.pt": blob}
        with ZipFile(stage, "x") as after:
            for entry in before.infolist():
                payload = payloads.get(entry.filename)
                after.writestr(copy(entry), payload if payload is not None else before.read(entry))
            if manas_zip:
                after.write(weights, "team_a.pt", compress_type=ZIP_DEFLATED)
        with ZipFile(stage) as check:
            if check.namelist() != names or check.testzip() is not None:
                raise ValueError("changed inventory or invalid candidate archive")
            for name in before.namelist():
                expected = payloads.get(name)
                if check.read(name) != (before.read(name) if expected is None else expected):
                    raise ValueError(f"unexpected changed entry: {name}")
            if manas_zip:
                with check.open("team_a.pt") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != weights_sha:
                        raise ValueError("changed MANAS bytes")
        os.link(stage, out)
    with out.open("rb") as stream:
        print(hashlib.file_digest(stream, "sha256").hexdigest(), out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("arm", choices=tuple(COUNTS)[1:])
    donor = p.add_mutually_exclusive_group(required=True)
    donor.add_argument("--extra", type=Path)
    donor.add_argument("--manas-zip", type=Path)
    p.add_argument("--third", type=Path)
    a = p.parse_args()
    archive(a.parent, a.out, a.arm, a.extra.read_bytes() if a.extra else None, a.manas_zip,
            a.third.read_bytes() if a.third else None)
