"""Verify uploaded byte archives and extract both into fresh local directories."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json


def sha(data):
    return hashlib.sha256(data).hexdigest()


reference = Path('/mnt/lustre/work/gehler/gfh527/ni26_native353_qualify_v1_20261117/package')
manifest = {}
for arm, archive, expected in [
    ('old', 'sealed_cand_v51b_own.zip', '4a0bc427f2c9c862ee779db5766badeae93f783446561f4dc8abb43589933bef'),
    ('cap', 'v66_cp1_cap250.zip', '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed'),
]:
    assert sha(Path(archive).read_bytes()) == expected
    folder = Path(arm)
    folder.mkdir()
    with ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(set(names)) == len(names) == (33 if arm == 'old' else 39)
        for name in names:
            assert name == Path(name).name
            data = z.read(name)
            if arm == 'cap':
                assert data == (reference / name).read_bytes()
            (folder / name).write_bytes(data)
            manifest[f'{arm}/{name}'] = sha(data)
Path('members.sha256').write_text(''.join(f'{s}  {n}\n' for n,s in manifest.items()))
