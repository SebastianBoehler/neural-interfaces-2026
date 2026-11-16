"""Chain jobs must select their chain workspace, not the submitting workspace."""

import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize('script', ['stage.sbatch', 'chain_dispatch.sbatch', 'member.sbatch', 'score.sbatch'])
def test_chain_job_enters_explicit_chain_workspace(tmp_path, script):
    producer, chain = tmp_path / 'producer', tmp_path / 'chain'
    producer.mkdir()
    chain.mkdir()
    lines = (HERE / script).read_text().splitlines()
    commands = [line for line in lines if line.startswith('cd ')]
    assert len(commands) == 1
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(producer), S308_CHAIN=str(chain))
    output = subprocess.check_output(['bash', '-c', commands[0] + '\npwd'], env=env, text=True)
    assert Path(output.strip()).resolve() == chain.resolve()
