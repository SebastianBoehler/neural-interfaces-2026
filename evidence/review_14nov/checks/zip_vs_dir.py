"""Does the staged ZIP hold exactly the package directory's files, and is its submission.py the current bundle?"""
import hashlib, sys, zipfile
from pathlib import Path
zp, d, bundle = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
want = dict(reversed(l.split()) for l in Path(f"{d}.sha256").read_text().splitlines())
with zipfile.ZipFile(zp) as z:
    got = {i.filename: hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist()}
print("zip sha256", hashlib.sha256(zp.read_bytes()).hexdigest())
print("members", len(got), "listed", len(want), "equal to the build's .sha256 list:", got == want)
print("submission.py == current bundle:", got.get("submission.py") == hashlib.sha256(bundle.read_bytes()).hexdigest())
print("extra / missing:", sorted(set(got) - set(want)), sorted(set(want) - set(got)))
