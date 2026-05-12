from pathlib import Path

import pytest
from slurm_compose.api import PyxisJobStep


@pytest.fixture
def pyxis_job_step() -> PyxisJobStep:
    return PyxisJobStep(
        job_name="test_step",
        container_image="/images/test.sqsh",
        container_mounts=["/var/run:/var/run"],
        nodes=2,
        ntasks_per_node=1,
        cpus_per_task=4,
        gpus_per_node=8,
        mem="16G",
        output=Path.home() / "slurm/test/%j.log",
        extra_argv=["--exact"],
        command=["python", "test.py"],
        env=dict(
            JOB_VAR="job_value",
        ),
    )


def test_pyxis_job_step_empty():
    with pytest.raises(ValueError):
        PyxisJobStep()


def test_pyxis_job_step(pyxis_job_step: PyxisJobStep):
    command = " ".join(pyxis_job_step.argv)

    assert f"--container-image {pyxis_job_step.container_image}" in command
    assert isinstance(pyxis_job_step.container_mounts, str)
    assert f"--container-mounts {pyxis_job_step.container_mounts}" in command
    assert "--no-container-mount-home" in command


def test_pyxis_job_step_nullables(pyxis_job_step: PyxisJobStep):
    pyxis_job_step.container_mounts = None
    pyxis_job_step.container_mount_home = True

    command = " ".join(pyxis_job_step.argv)

    assert "--container-mounts" not in command
    assert "--container-mount-home" in command


def test_pyxis_job_step_invalid(pyxis_job_step: PyxisJobStep):
    pyxis_job_step.container_mounts = []

    with pytest.raises(ValueError):
        pyxis_job_step.argv
