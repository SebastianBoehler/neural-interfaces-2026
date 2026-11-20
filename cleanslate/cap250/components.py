"""Sealed cap250 member layout and fold package switches."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Family:
    seeds: int
    minutes: int
    gpu: bool = True
    cpus: int = 4
    memory: str = "16G"


FAMILIES = {
    "shallow_ea": Family(5, 12),
    "shallow_raw": Family(5, 12),
    "fbc_ea_fixed": Family(5, 10),
    "fbc_raw_fixed": Family(5, 10),
    "fbc_ea_fir": Family(2, 10),
    "fbc_raw_fir": Family(2, 10),
    "sinc_ea": Family(5, 12),
    "fbl_ea": Family(5, 10),
    "sbn_fbc_ea": Family(6, 12),
    "sbn_fbc_raw": Family(6, 10),
    "sbn_sinc": Family(6, 12),
    "sbn_fbl": Family(6, 10),
    "teacher": Family(1, 60, cpus=8, memory="48G"),
    "eye": Family(3, 10),
    "rest": Family(3, 12),
    "geometry": Family(1, 65, False, cpus=8, memory="48G"),
    "hilbert": Family(1, 35, False, cpus=8, memory="32G"),
    "identity": Family(1, 40, False, cpus=8, memory="24G"),
    "count": Family(1, 20, False, cpus=4),
    "hard": Family(2, 20),
    "cbramod": Family(3, 25, cpus=8, memory="40G"),
}

ORDINARY = dict(zip(tuple(FAMILIES)[:8], (0, 5, 10, 15, 20, 22, 24, 29), strict=True))
SBN = {
    "sbn_fbc_ea": ((35, "sbn_k2"), (41, "sbn_k5")),
    "sbn_fbc_raw": ((47, "sbn_fbcnet_raw_k20"),),
    "sbn_sinc": ((53, "sbn_sincshallow_ea_ptm7r_rot05_k20"),),
    "sbn_fbl": ((59, "sbn_fblight_ea_rot05_k5"),),
}
TTT_FORMS = ("stock", "off", "tttlite5", "tttsparse20", "tttlight", "tttheavy")


@dataclass(frozen=True)
class Variant:
    hard: bool = False
    cb: bool = False
    ttt: str = "stock"

    @classmethod
    def from_env(cls):
        hard = os.environ.get("CAP250_HARD", "0")
        cb = os.environ.get("CAP250_CB", "0")
        ttt = os.environ.get("CAP250_TTT", "stock")
        if hard not in ("0", "1") or cb not in ("0", "1") or ttt not in TTT_FORMS:
            raise ValueError(f"invalid cap250 variant: HARD={hard}, CB={cb}, TTT={ttt}")
        return cls(hard == "1", cb == "1", ttt)

    def env(self):
        return {"CAP250_HARD": str(int(self.hard)), "CAP250_CB": str(int(self.cb)), "CAP250_TTT": self.ttt}

    def fits(self):
        optional = {"hard": self.hard, "cbramod": self.cb}
        return [(name, seed) for name, family in FAMILIES.items() if optional.get(name, True)
                for seed in range(family.seeds)]


def part_path(parts, family, seed):
    return parts / family / f"s{seed}"


def assets(family, seed):
    if family == "teacher":
        return ("teacher.pt",)
    if family in SBN:
        return tuple(f"{prefix}_s{seed}.pt" for _, prefix in SBN[family])
    if family == "eye":
        return (f"eye_s{seed}.pt",)
    if family == "rest":
        return (f"rest_s{seed}.pt", f"rest_s{seed}.npz")
    if family == "hard":
        return (f"hard_s1c_s{seed}.pt",)
    if family == "cbramod":
        return (f"cb_s{seed}.pt",)
    if family == "count":
        return ("cf_params.npz",)
    if family == "geometry":
        return ("tpri.npz",)
    return ()


def expected_files(family, seed):
    files = assets(family, seed)
    if family in ORDINARY or family in SBN or family in ("teacher", "geometry", "hilbert", "identity"):
        return ("weights.npz", *files)
    return files
