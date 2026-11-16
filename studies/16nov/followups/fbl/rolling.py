"""Use the unchanged §298 collector for one complete §299 dose."""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    registration = ROOT / 'runs/lane1_inputs/fbl-registration.md'
    if not registration.is_file() or '### 299.' not in registration.read_text():
        raise ValueError('§299 registration required')
    namespace = runpy.run_path(str(HERE.parent / 'system/rolling_v2.py'), run_name='fbl_rolling')
    original = namespace['record']
    def record(wave, arm, paths, prereg, digest):
        return original('299', arm, [*paths, registration], prereg, digest)
    namespace['main'].__globals__['record'] = record
    namespace['main']()


if __name__ == '__main__': main()
