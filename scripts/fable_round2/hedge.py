"""The preregistered P3 schedule applied to the lead's Team B mixer."""
from pathlib import Path

SELECT = tuple(range(71180, 71188))
CONFIRM = tuple(range(71188, 71196))
LEAD_ARM = "arith25"
CONTROL = "log25"
GRID = {
    "default": (0.60, 0.15, 30, 0.30),
    "lower_start": (0.40, 0.15, 30, 0.30),
    "higher_floor": (0.60, 0.25, 30, 0.30),
    "slower": (0.60, 0.15, 60, 0.30),
    "less_disagreement": (0.60, 0.15, 30, 0.10),
    "damped": (0.36, 0.17, 30, 0.12),
    "late_ramp": (0.15, 0.24, 30, 0.10),
}
SCHEDULES = {LEAD_ARM: None, CONTROL: (0.25, 0.25, 30, 0.0), **GRID}


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected one exact lead source block for {old!r}")
    return source.replace(old, new, 1)


def revised(source, schedule=None):
    declaration = "    TEAM_B_MEMBERS = 3\n"
    source = once(source, declaration, declaration + f"    ROUND2_HEDGE = {schedule!r}\n")
    reset = "        if self.team_b is not None:\n            self.team_b.reset()\n"
    source = once(source, reset, reset + "        self.hedge_seen, self.hedge_agreed = 0, 0\n")
    blend = "        return (1 - self.TEAM_B_MIX) * m + self.TEAM_B_MIX * q\n"
    method = Path(__file__).with_name("hedge_pool.py.inc").read_text()
    source = once(source, blend, "        if self.ROUND2_HEDGE is None:\n    " + blend +
                  "        return self.hedge_pool(m, q)\n\n" + method.rstrip() + "\n")
    compile(source, "submission.py", "exec")
    return source
