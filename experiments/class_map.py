"""Explicit class-name -> task-index map for every external corpus row (fail loudly; 20 Nov audit, bugs_datapath 3).

Task indices are the PROTEUS targets: 0 motor imagery, 1 subtraction, 2 word generation. Rest / baseline carry no task
class: ``AUX`` marks them and ``task_labels`` returns -1 for them, so a caller must drop or route them on purpose.
``protocol.load`` used to copy the file's ``y`` unchanged, so shin2018 (``baseline=0, WG=1``) trained as MI / SUB."""
import numpy as np

TASK = {"left_hand": 0, "right_hand": 0, "hands": 0, "feet": 0, "subtraction": 1, "WG": 2, "S16": 2}
AUX = {"rest", "baseline", "S32"}
CLASSLESS = {"scherer2015": ("hands", "subtraction", "S16")}  # export file has no `classes`: HAND 0, SUB 1, WORD 2 (laneC)
EXPECT = {"shin2017a": {"left_hand", "right_hand"}, "shin2017b": {"rest", "subtraction"}, "shin2018": {"baseline", "WG"}}


def class_names(name, f):
    """Class names of an external npz ``f`` (a loaded NpzFile or dict), pinned for classless exports."""
    names = [str(c) for c in f["classes"]] if "classes" in f else list(CLASSLESS.get(name, ()))
    assert names, f"{name}: no class names in the file and no pinned CLASSLESS entry; refusing to guess indices"
    assert name not in EXPECT or set(names) == EXPECT[name], f"{name}: classes {names} != expected {sorted(EXPECT[name])}"
    return names


def task_labels(name, names, y):
    """Map a corpus-local index vector ``y`` to task indices (0/1/2), or -1 for AUX; unknown names raise."""
    unknown = [c for c in names if c not in TASK and c not in AUX]
    assert not unknown, f"{name}: unmapped class name(s) {unknown}; add them to class_map.TASK/AUX on purpose"
    y = np.asarray(y)
    assert y.min() >= 0 and y.max() < len(names), f"{name}: y outside the {len(names)} named classes"
    lut = np.array([TASK.get(c, -1) for c in names])
    return lut[y]
