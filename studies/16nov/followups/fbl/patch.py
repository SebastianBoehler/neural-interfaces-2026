"""Restrict a private §294 recipe copy to the preregistered §299 doses."""
import hashlib
import sys
from pathlib import Path

EXPECTED = '71571cc288069185fb15f95ffd9bf0ba34d88273cff6a822180ad91b4463e9d1'
FIXED = """ARMS = {
    'FBL05': ('none', 0., 'none', .5),
    'FBL15': ('none', 0., 'none', 1.5),
    'FBL2': ('none', 0., 'none', 2.),
    'FBL3': ('none', 0., 'none', 3.),
}
"""


def patch(source):
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED:
        raise ValueError('source recipe is not the pinned §294 version')
    start, end = source.index('ARMS = {'), source.index('ZERO =')
    return source[:start] + FIXED + source[end:]


if __name__ == '__main__':
    path = Path(sys.argv[1]); before = path.read_text(); after = patch(before)
    path.write_text(after)
