import pytest
from slurm_compose.api.scripts import PyxisScript
from slurm_compose.api.scripts.utils import maybe_update_fields


@pytest.fixture
def pyxis_job_step() -> PyxisScript:
    return PyxisScript(job_name="test_step", container_image="/var/images/container.sqsh", command=["true"])


def test_update_fields(pyxis_job_step: PyxisScript):
    maybe_update_fields(pyxis_job_step, job_name="new_test_step", cpus_per_task=32)

    assert pyxis_job_step.job_name == "test_step"
    assert pyxis_job_step.cpus_per_task == 32

    maybe_update_fields(pyxis_job_step, job_name="new_test_step", cpus_per_task=16, force=True)
    assert pyxis_job_step.job_name == "new_test_step"
    assert pyxis_job_step.cpus_per_task == 16


def test_update_gpus_per_node(pyxis_job_step: PyxisScript):
    maybe_update_fields(pyxis_job_step, gpus_per_node=8)

    assert pyxis_job_step.gpus_per_node == 8


def test_update_gpus_per_node_when_zero(pyxis_job_step: PyxisScript):
    pyxis_job_step.gpus_per_node = 0

    maybe_update_fields(pyxis_job_step, gpus_per_node=8)
    assert pyxis_job_step.gpus_per_node == 0

    maybe_update_fields(pyxis_job_step, gpus_per_node=8, force=True)
    assert pyxis_job_step.gpus_per_node == 0

    maybe_update_fields(pyxis_job_step, gpus_per_node=-1, force=True)
    assert pyxis_job_step.gpus_per_node == 0
