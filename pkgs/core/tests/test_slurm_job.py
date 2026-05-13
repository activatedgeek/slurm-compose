from datetime import timedelta
from pathlib import Path

import pytest
from slurm_compose.api import SlurmJob, SlurmJobStep, SrunJobStep


@pytest.fixture
def slurm_job_step() -> SlurmJobStep:
    return SlurmJobStep(
        command="sleep infinity",
        env=dict(
            TEST_VAR="test_value",
        ),
    )


@pytest.fixture
def srun_job_step() -> SrunJobStep:
    return SrunJobStep(
        job_name="test_step",
        nodes=2,
        ntasks_per_node=1,
        cpus_per_task=4,
        gpus_per_node=8,
        mem="16G",
        output=Path.home() / "slurm/test/%j.log",
        command=["python", "test.py"],
        env=dict(
            JOB_VAR="job_value",
        ),
    )


@pytest.fixture
def slurm_job() -> SlurmJob:
    return SlurmJob(
        job_name="test_job",
        account="test_account",
        partition="test_partition",
        qos="test_qos",
        time=timedelta(hours=4),
        mem="16G",
        cpus_per_task=1,
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
    assert f"#SBATCH --open-mode={slurm_job.open_mode}\n" in materialized_str


def test_slurm_job_materialize_nullables(slurm_job: SlurmJob):
    slurm_job.qos = None
    slurm_job.mem = None
    slurm_job.cpus_per_task = None
    slurm_job.gpus_per_node = None
    slurm_job.output = None
    slurm_job.error = None

    materialized_str = slurm_job.materialize()

    assert "#SBATCH --qos=" not in materialized_str
    assert "#SBATCH --mem=" not in materialized_str
    assert "#SBATCH --cpus-per-task=" not in materialized_str
    assert "#SBATCH --gpus-per-node=" not in materialized_str
    assert "#SBATCH --output=" not in materialized_str
    assert "#SBATCH --error=" not in materialized_str


def test_slurm_job_extras_materialize(slurm_job: SlurmJob):
    slurm_job.extra_argv = ["--comment=test_comment"]

    materialized_str = slurm_job.materialize()

    assert "#SBATCH --comment=test_comment\n" in materialized_str


def test_job_step_empty():
    with pytest.raises(ValueError):
        SlurmJobStep()

    with pytest.raises(ValueError):
        SrunJobStep()


def test_slurm_job_step(slurm_job_step: SlurmJobStep):
    assert len(slurm_job_step.argv) == 2


def test_srun_job_step(srun_job_step: SrunJobStep):
    command = srun_job_step.args

    assert command.startswith("srun")
    assert f"--job-name {srun_job_step.job_name}" in command
    assert f"--nodes {srun_job_step.nodes}" in command
    assert f"--ntasks-per-node {srun_job_step.ntasks_per_node}" in command
    assert f"--cpus-per-task {srun_job_step.cpus_per_task}" in command
    assert f"--gpus-per-node {srun_job_step.gpus_per_node}" in command
    assert f"--mem {srun_job_step.mem}" in command
    assert f"--output {srun_job_step.output}" in command
    assert f"--error {srun_job_step.error}" in command
    assert f"--wait {srun_job_step.wait}" in command
    assert f"--kill-on-bad-exit {srun_job_step.kill_on_bad_exit}" in command
    assert "--exact" in command
    assert "--overlap" not in command
    assert f"{' '.join(srun_job_step.extra_argv)}" in command


def test_srun_job_step_nullables(srun_job_step: SrunJobStep):
    srun_job_step.cpus_per_task = None
    srun_job_step.mem = None
    srun_job_step.exact = None
    srun_job_step.overlap = True

    command = srun_job_step.args

    assert "--cpus-per-task" not in command
    assert "--mem" not in command
    assert "--exact" not in command
    assert "--overlap" in command
