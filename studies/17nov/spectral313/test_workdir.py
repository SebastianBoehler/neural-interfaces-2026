import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize('script', ['stage.sbatch', 'member.sbatch', 'score_chain.sbatch', 'launch.sbatch'])
def test_chain_jobs_enter_explicit_chain_directory(tmp_path, script):
    producer, chain = tmp_path / 'producer', tmp_path / 'chain'
    producer.mkdir()
    chain.mkdir()
    commands = [line for line in (HERE / script).read_text().splitlines() if line.startswith('cd ')]
    assert len(commands) == 1
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(producer), S313_CHAIN=str(chain))
    output = subprocess.check_output(['bash', '-c', commands[0] + '\npwd'], env=env, text=True)
    assert Path(output.strip()).resolve() == chain.resolve()


@pytest.mark.parametrize('script', ['collect.sbatch', 'score_whole.sbatch'])
def test_collector_jobs_enter_explicit_collector_directory(tmp_path, script):
    producer, collector = tmp_path / 'producer', tmp_path / 'collector'
    producer.mkdir()
    collector.mkdir()
    commands = [line for line in (HERE / script).read_text().splitlines() if line.startswith('cd ')]
    assert len(commands) == 1
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(producer), S313_COLLECT_WS=str(collector))
    output = subprocess.check_output(['bash', '-c', commands[0] + '\npwd'], env=env, text=True)
    assert Path(output.strip()).resolve() == collector.resolve()
