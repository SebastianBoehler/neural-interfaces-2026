"""Run the original all10 command after independently re-verifying every active proof."""
import os
import shutil
import sys
from pathlib import Path

from proofs import ACTIVE,manifest,verify
import all10


def main(seed):
    ws=Path.cwd();manifest(ws);verify()
    folder=ws/f'seed{seed}';folder.mkdir(exist_ok=False)
    for name in ('result.json','outer_active_path.json','base_v66.zip','build.json'):
        shutil.copy2(ACTIVE/name,folder/name)
    for path in ACTIVE.glob('study_recipe_*.json'):shutil.copy2(path,folder/path.name)
    for name in ('draft_jsall','export_external_teacher.py','source.sha256'):
        (folder/name).symlink_to((ACTIVE if name!='source.sha256' else ws)/name)
    os.chdir(folder);all10.main(seed)


if __name__=='__main__':main(int(sys.argv[1]))
