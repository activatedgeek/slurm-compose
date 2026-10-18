import pytest
from pydantic import ValidationError

from slurm_compose.taskfile import BashRunner, PyxisKwargs, PyxisRunner, Runner, SrunRunner, Task, Taskfile


def test_default_runner_is_bash():
    task = Task(cmds=["echo hello"])
    assert isinstance(task.runner, BashRunner)
    assert "runner" not in task.model_dump(exclude_defaults=True)


def test_bash_returns_commands_unchanged():
    cmds = ["echo hello", "echo world"]
    assert BashRunner()(cmds) is cmds


@pytest.mark.parametrize(
    "runner",
    [Runner(type="base"), SrunRunner(), PyxisRunner(kwargs=PyxisKwargs(container_image="model.sqsh"))],
)
def test_unimplemented_runner_transform(runner):
    with pytest.raises(NotImplementedError):
        runner(["echo hello"])


@pytest.mark.parametrize(
    ("runner", "expected"),
    [
        ({"type": "bash"}, BashRunner),
        ({"type": "srun"}, SrunRunner),
        ({"type": "srun", "kwargs": {"nodes": 2, "gpus_per_node": 0}}, SrunRunner),
        (
            {"type": "pyxis", "kwargs": {"container_image": "/shared/model.sqsh", "gpus_per_node": 1}},
            PyxisRunner,
        ),
    ],
)
def test_runner_selection_and_round_trip(runner, expected):
    workflow = Taskfile.model_validate({"version": "3", "tasks": {"run": {"runner": runner}}})
    assert isinstance(workflow.tasks["run"].runner, expected)
    assert Taskfile.model_validate_json(workflow.model_dump_json()) == workflow


@pytest.mark.parametrize(
    "runner",
    [
        {"type": "unknown"},
        {"type": "bash", "kwargs": {"nodes": 1}},
        {"type": "srun", "kwargs": {"nodes": 0}},
        {"type": "srun", "kwargs": {"nodes": "2"}},
        {"type": "srun", "kwargs": {"gpus_per_node": -1}},
        {"type": "srun", "kwargs": {"unknown": True}},
        {"type": "pyxis", "kwargs": {}},
        {"type": "pyxis", "kwargs": {"container_image": " "}},
    ],
)
def test_reject_invalid_runner(runner):
    with pytest.raises(ValidationError):
        Task.model_validate({"runner": runner})
