"""Render the registered review from pinned labels and optional frozen captures."""
import argparse
from pathlib import Path

import numpy as np

from errors import error_summary
from tapes import labels, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--part", choices=("error", "stages", "fit", "held"), default="error")
    a = p.parse_args()
    data = metadata(a.data)
    interfaces = tuple(sorted(np.unique(data["interface"])))
    orders = tuple(range(71800, 71816))
    if a.part == "error":
        text = error_summary(labels(data, a.work / "leo", orders), orders, interfaces)
    else:
        from diagnostics import component
        text = component(a.work, data, interfaces, a.part)
    a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
