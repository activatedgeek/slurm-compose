from datetime import timedelta
from pathlib import Path

import pytest
from slurm_compose.api import SlurmJob


@pytest.fixture
def slurm_job() -> SlurmJob:
    return SlurmJob(
        job_name="test_job",
        account="test_account",
        partition="test_partition",
        qos="test_qos",
        time=timedelta(hours=4),
        mem="16G",
        gpus_per_node=8,
        output=Path.home() / "slurm/test/%j.log",
    )


def test_slurm_job_materialize(slurm_job: SlurmJob):
    materialized_str = slurm_job.materialize()

    assert f"#SBATCH --job-name={slurm_job.job_name}\n" in materialized_str
    assert f"#SBATCH --account={slurm_job.account}\n" in materialized_str
    assert f"#SBATCH --partition={slurm_job.partition}\n" in materialized_str
    assert f"#SBATCH --qos={slurm_job.qos}\n" in materialized_str
    assert "#SBATCH --time=04:00:00\n" in materialized_str
    assert f"#SBATCH --nodes={slurm_job.nodes}\n" in materialized_str
    assert f"#SBATCH --mem={slurm_job.mem}\n" in materialized_str
    assert f"#SBATCH --cpus-per-task={slurm_job.cpus_per_task}\n" in materialized_str
    assert f"#SBATCH --gpus-per-node={slurm_job.gpus_per_node}\n" in materialized_str
    assert f"#SBATCH --ntasks-per-node={slurm_job.ntasks_per_node}\n" in materialized_str
    assert f"#SBATCH --output={slurm_job.output}\n" in materialized_str
    assert f"#SBATCH --error={slurm_job.error}\n" in materialized_str


def test_slurm_job_nullable_materialize(slurm_job: SlurmJob):
    slurm_job.qos = None
    slurm_job.mem = None
    slurm_job.gpus_per_node = None
    slurm_job.output = None
    slurm_job.error = None

    materialized_str = slurm_job.materialize()

    assert "#SBATCH --qos=" not in materialized_str
    assert "#SBATCH --mem=" not in materialized_str
    assert "#SBATCH --gpus-per-node=" not in materialized_str
    assert "#SBATCH --output=" not in materialized_str
    assert "#SBATCH --error=" not in materialized_str


def test_slurm_job_extras_materialize(slurm_job: SlurmJob):
    slurm_job.extras = ["--comment=test_comment"]

    materialized_str = slurm_job.materialize()

    assert "#SBATCH --comment=test_comment\n" in materialized_str
