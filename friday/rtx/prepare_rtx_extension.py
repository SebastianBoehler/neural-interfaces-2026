"""Stage the verified D-RTX worker with its participant guard extended."""

from pathlib import Path
import sys


def main(source, destination):
    original, extended = Path(source), Path(destination)
    value = original.read_text()
    old = '    assert (person, who) in ((0, "Scherer2015/A"), (5, "Scherer2015/G"))\n'
    new = "    assert len(subjects) == 9 and 0 <= person < len(subjects)\n"
    if value.count(old) != 1:
        raise ValueError("expected the pinned two-person D-RTX guard")
    extended.write_text(value.replace(old, new))


if __name__ == "__main__":
    main(*sys.argv[1:])
