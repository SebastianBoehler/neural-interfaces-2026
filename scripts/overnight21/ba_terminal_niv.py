"""Stream the Scherer-free hybrid, optionally overlaying the matched visit-1 enrollment."""
from hashlib import file_digest
import os
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path.cwd() / 'niv'))
import worker

CANDIDATE_SHA = '4143ac070dbedb4b4962ab734407caed4629b5607af49b5e547c8041ed65c682'


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def main(out, person, arms, enrolled_sha=''):
    archive = '/candidate.zip'
    if digest(archive) != CANDIDATE_SHA:
        raise ValueError('NI-V candidate ZIP differs')
    if enrolled_sha:
        visits = os.environ.get('NIV_VISITS')
        enrolled = Path('/enrolled.zip')
        if visits != '2' or digest(enrolled) != enrolled_sha:
            raise ValueError('visit-1 enrollment receipt or test visit differs')
        original = worker.load

        def load(directory, channels):
            # Reuse exactly the baseline person's enrolled base. The MANAS bank remains frozen.
            with ZipFile(enrolled) as source:
                (directory / 'weights.npz').write_bytes(source.read('weights.npz'))
            return original(directory, channels)

        worker.load = load
    worker.main(archive, out, person, arms)


if __name__ == '__main__':
    main(*sys.argv[1:])
