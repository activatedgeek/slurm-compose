import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from slurm_compose.cli import ConfigCommand, UpCommand
from slurm_compose.config import STATE_HOME
from slurm_compose.legacy_cli import CLIConfig


@pytest.fixture
def cli_config() -> CLIConfig:
    return CLIConfig(file=Path(__file__).parent / "configs/test.yml", nodes=2)


@pytest.fixture
def cli_wait_config() -> CLIConfig:
    return CLIConfig(file=Path(__file__).parent / "configs/wait.yml", delay=0)


@pytest.mark.skip(reason="legacy CLI")
def test_cli(cli_config: CLIConfig):
    with TemporaryDirectory() as tmp:
        cli_config.export_dir = tmp

        cli_config.run()

        for export in cli_config.exports:
            assert export.job.nodes == 2
            assert export.job.time == "01:00:00"
            assert export.sbatch_file.exists()
            assert os.access(export.sbatch_file, os.X_OK)

            assert (export.package_dir / "slurm_compose").is_dir()


@pytest.mark.skip(reason="legacy CLI")
def test_cli_wait_exec(cli_wait_config: CLIConfig):
    with TemporaryDirectory() as tmp:
        cli_wait_config.export_dir = tmp

        cli_wait_config.run()

        ok_job = cli_wait_config.exports[0].job
        assert ok_job.job_name.startswith("ok.")

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

        fail_job = cli_wait_config.exports[1].job
        assert fail_job.job_name.startswith("fail.")

        fail_job_result = subprocess.run(
            [cli_wait_config.exports[1].sbatch_file],
            capture_output=True,
            text=True,
            check=False,
        )
        assert fail_job_result.returncode == 1
        assert "TESTENV: testvalue" in fail_job_result.stdout
    assert "Finished Step" not in fail_job_result.stdout


def test_new_cli_loads_compose_config():
    command = ConfigCommand(file=Path(__file__).parent / "configs/spec.yml")
    command()
    config = command.compose
    assert config.name == "evaluation"
    assert config.directory == STATE_HOME


def test_new_cli_overrides_compose_config():
    cli_config = ConfigCommand(
        file=Path(__file__).parent / "configs/spec.yml",
        job_name="cli-evaluation",
        jobs_dir=Path("/tmp/cli-evaluation"),
    )
    cli_config()
    config = cli_config.compose
    assert config.name == "cli-evaluation"
    assert config.directory == Path("/tmp/cli-evaluation")


def test_new_cli_overrides_compose_vars():
    command = ConfigCommand(
        file=Path(__file__).parent / "configs/spec.yml",
        vars=["MODEL=llama", "NODES=2"],
    )
    assert command.compose.vars["MODEL"] == "llama"
    assert command.compose.vars["NODES"] == 2
    assert command.compose.steps["model"].command == ["/shared/project/start-model.sh", "--model", "llama"]


def test_new_cli_up_stores_passthrough_arguments(capsys):
    command = UpCommand(
        file=Path(__file__).parent / "configs/spec.yml",
        passthrough=["--partition=gpu", "--comment", "evaluation"],
    )
    assert command.passthrough == ["--partition=gpu", "--comment", "evaluation"]
    command()
    output = capsys.readouterr().out
    assert "#!/usr/bin/env bash" in output
    assert "#SBATCH --comment\n" not in output
    assert "#SBATCH evaluation\n" not in output
