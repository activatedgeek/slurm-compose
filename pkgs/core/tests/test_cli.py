import os
import subprocess
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
        cli_config.export_dir = tmp

        cli_config.run()

        for export in cli_config.exports:
            assert export.sbatch_file.is_file()
            assert os.access(export.sbatch_file, os.X_OK)

            assert (export.package_dir / "slurm_compose").is_dir()


def test_cli_wait_exec(cli_wait_config: CLIConfig):
    with TemporaryDirectory() as tmp:
        cli_wait_config.export_dir = tmp

        cli_wait_config.run()

        ok_job = cli_wait_config.jobs[0]
        assert ok_job.job_name == "ok"

        ok_job_result = subprocess.run(
            [cli_wait_config.exports[0].sbatch_file],
            capture_output=True,
            text=True,
            check=False,
        )
        assert ok_job_result.returncode == 0
        assert "Finished Step 0." not in ok_job_result.stdout
        assert "Finished Step 1." in ok_job_result.stdout
        assert "Finished Step 2." not in ok_job_result.stdout

        fail_job = cli_wait_config.jobs[1]
        assert fail_job.job_name == "fail"

        fail_job_result = subprocess.run(
            [cli_wait_config.exports[1].sbatch_file],
            capture_output=True,
            text=True,
            check=False,
        )
        assert fail_job_result.returncode == 1
        assert "TESTENV: testvalue" in fail_job_result.stdout
        assert "Finished Step" not in fail_job_result.stdout
