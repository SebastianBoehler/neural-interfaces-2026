import os
import subprocess
from pathlib import Path


def test_bootstrap_uses_existing_environment_without_optional_module(tmp_path):
    job = Path(__file__).resolve().parents[1] / "job.sh"
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/activate").write_text("")
    (tmp_path / "source.sha256").write_text("")
    shell_env = tmp_path / "shell_env"
    shell_env.write_text("module() { return 1; }\nsha256sum() { return 0; }\npython() { echo PYTHON_CALLED; }\n")
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(tmp_path), BASH_ENV=str(shell_env))
    result = subprocess.run(["bash", str(job), "smoke"], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("PYTHON_CALLED") == 4
