import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from slurm_compose.cli import CLIConfig


@pytest.fixture
def cli_config() -> CLIConfig:
    return CLIConfig(file="pkgs/core/tests/test_config.yml")


@pytest.fixture
def cli_wait_config() -> CLIConfig:
    return CLIConfig(file="pkgs/core/tests/test_wait.yml")


def test_cli(cli_config: CLIConfig):
    with TemporaryDirectory() as tmp:
        cli_config.write = tmp

        cli_config.run()

        for job in cli_config.jobs:
            sbatch_file = Path(tmp) / f"{job.job_name}_sbatch.sh"
            assert sbatch_file.is_file()
            assert os.access(sbatch_file, os.X_OK)


def test_cli_wait_exec(cli_wait_config: CLIConfig):
    with TemporaryDirectory() as tmp:
        cli_wait_config.write = tmp

        cli_wait_config.run()

        ok_job = cli_wait_config.jobs[0]
        assert ok_job.job_name == "ok"

        ok_job_result = subprocess.run(
            [Path(tmp) / f"{ok_job.job_name}_sbatch.sh"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert ok_job_result.returncode == 0
        assert "Finished Step 1." not in ok_job_result.stdout
        assert "Finished Step 2." in ok_job_result.stdout
        assert "Finished Step 3." not in ok_job_result.stdout

        fail_job = cli_wait_config.jobs[1]
        assert fail_job.job_name == "fail"

        fail_job_result = subprocess.run(
            [Path(tmp) / f"{fail_job.job_name}_sbatch.sh"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert fail_job_result.returncode == 1
        assert "TESTENV: testvalue" in fail_job_result.stdout
        assert "Finished Step" not in fail_job_result.stdout
