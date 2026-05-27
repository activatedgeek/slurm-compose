from pathlib import Path

import pytest
from slurm_compose.api.scripts import PyxisScript


@pytest.fixture
def pyxis_job_step() -> PyxisScript:
    return PyxisScript(
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
        PyxisScript()


def test_pyxis_job_step(pyxis_job_step: PyxisScript):
    command = pyxis_job_step.args

    assert f"--container-image {pyxis_job_step.container_image}" in command
    assert isinstance(pyxis_job_step.container_mounts, list)
    assert f"--container-mounts {','.join(pyxis_job_step.container_mounts)}" in command
    assert "--no-container-mount-home" in command


def test_pyxis_job_step_nullables(pyxis_job_step: PyxisScript):
    pyxis_job_step.container_mounts = None
    pyxis_job_step.container_mount_home = True

    command = pyxis_job_step.args

    assert "--container-mounts" not in command
    assert "--container-mount-home" in command
