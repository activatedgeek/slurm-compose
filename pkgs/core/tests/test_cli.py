import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from slurm_compose.cli import CLIConfig


@pytest.fixture
def cli_config() -> CLIConfig:
    return CLIConfig(file="pkgs/core/tests/test_config.yml")


def test_cli(cli_config: CLIConfig):
    with TemporaryDirectory() as tmp:
        cli_config.write = tmp

        cli_config.run()

        for job in cli_config.jobs:
            sbatch_file = Path(tmp) / f"{job.job_name}_sbatch.sh"
            assert sbatch_file.is_file()
            assert os.access(sbatch_file, os.X_OK)
