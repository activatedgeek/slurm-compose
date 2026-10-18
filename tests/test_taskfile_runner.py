import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from slurm_compose.taskfile import BashRunner, PyxisKwargs, PyxisRunner, Runner, SrunKwargs, SrunRunner, Task, Taskfile


def test_default_runner_is_bash():
    task = Task(cmds=["echo hello"])
    assert isinstance(task.runner, BashRunner)
    assert "runner" not in task.model_dump(exclude_defaults=True)


def test_bash_returns_commands_unchanged():
    cmds = ["echo hello", "echo world"]
    assert BashRunner()(cmds) is cmds
    assert BashRunner().apply_defaults("train", Path("/tmp/run")) is None


@pytest.mark.parametrize(
    "runner",
    [Runner(type="base")],
)
def test_unimplemented_runner_transform(runner):
    with pytest.raises(NotImplementedError):
        runner(["echo hello"])


def test_srun_wraps_each_command_with_resources():
    runner = SrunRunner(
        kwargs=SrunKwargs(
            nodes=2,
            ntasks_per_node=1,
            cpus_per_task=4,
            gpus_per_node=0,
            mem="8G",
            extra_argv=["--exclusive", "--comment=a job; $(echo literal)"],
        )
    )
    cmds = ["echo one", "bash <<'BASH'\necho \"it's $MODEL\"\nBASH\n"]
    transformed = runner(cmds)
    for source, command in zip(cmds, transformed, strict=True):
        header, body = command.split("\n", 1)
        assert header.endswith(" <<'TASK_SRUN'")
        assert body == source.rstrip("\n") + "\nTASK_SRUN\n"
        assert (
            shlex.split(header.removesuffix(" <<'TASK_SRUN'"))
            == runner.pre_cmd
            == [
                "srun",
                "--nodes=2",
                "--ntasks-per-node=1",
                "--cpus-per-task=4",
                "--gpus-per-node=0",
                "--mem=8G",
                "--open-mode=append",
                "--exclusive",
                "--comment=a job; $(echo literal)",
                "bash",
                "-euo",
                "pipefail",
                "-s",
            ]
        )
    assert cmds[0] == "echo one"
    assert SrunRunner()([]) == []
    assert SrunRunner()(["echo hello"])[0] == (
        "srun --open-mode=append bash -euo pipefail -s <<'TASK_SRUN'\necho hello\nTASK_SRUN\n"
    )


@pytest.mark.parametrize(
    "runner",
    [
        SrunRunner(kwargs=SrunKwargs(nodes=1)),
        PyxisRunner(kwargs=PyxisKwargs(nodes=1, container_image="model.sqsh")),
    ],
)
def test_srun_executes_shell_syntax_and_preserves_failure(tmp_path, runner):
    # Stand in for Slurm to verify shell boundaries without a cluster.
    fake_srun = tmp_path / "srun"
    fake_srun.write_text('#!/bin/sh\nwhile [ "$1" != bash ]; do shift; done\nexec "$@"\n')
    fake_srun.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}", "MODEL": "it's qwen"}
    delimiter = runner.delimiter
    cmds = [
        'bash <<\'BASH\'\nprintf "%s\\n" "$MODEL"\nBASH\n',
        'printf "%s\\n" "$(printf second)" | cat',
        "exit 7",
        "false; echo should-not-run",
        "false | cat",
        'echo "$STASK_TEST_UNSET_VARIABLE"',
        "set +e; false; echo allowed",
        "set +o pipefail; false | cat",
        'set +u; echo "${STASK_TEST_UNSET_VARIABLE}"',
        f"cat <<'INNER'\n{delimiter}\n{delimiter}_1\nINNER\n",
    ]
    results = [
        subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True, check=False)
        for cmd in runner(cmds)
    ]
    assert [result.returncode for result in results] == [0, 0, 7, 1, 1, 1, 0, 0, 0, 0]
    assert results[0].stdout == "it's qwen\n"
    assert results[1].stdout == "second\n"
    assert results[3].stdout == ""
    assert results[6].stdout == "allowed\n"
    assert results[9].stdout == f"{delimiter}\n{delimiter}_1\n"
    assert f"<<'{delimiter}_2'" in runner([cmds[9]])[0]

    # Also exercise Task's embedded shell as the heredoc producer.
    workflow = Taskfile(version="3", tasks={"default": Task(runner=runner, cmds=[cmds[0], cmds[9]])})
    generated = tmp_path / "Taskfile.yml"
    generated.write_text(workflow.to_yaml(tmp_path))
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("task")),
            "--silent",
            "--dir",
            str(tmp_path),
            "--taskfile",
            str(generated),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"it's qwen\n{delimiter}\n{delimiter}_1\n"


@pytest.mark.parametrize("mount_home", [True, False])
def test_pyxis_extends_srun_options(mount_home):
    runner = PyxisRunner(
        kwargs=PyxisKwargs(
            nodes=1,
            container_image="/shared/model image.sqsh",
            container_mounts=["/shared/data:/data:ro", "/shared/results:/results"],
            container_workdir="/work dir",
            container_mount_home=mount_home,
        )
    )
    assert isinstance(runner, SrunRunner)
    assert runner.delimiter == "TASK_PYXIS"
    assert "delimiter" not in runner.model_dump()
    assert "<<'TASK_PYXIS'" in runner(["echo hello"])[0]
    assert runner.pre_cmd == [
        "srun",
        "--nodes=1",
        "--open-mode=append",
        "--container-image=/shared/model image.sqsh",
        "--container-mounts=/shared/data:/data:ro,/shared/results:/results",
        "--container-workdir=/work dir",
        "--container-mount-home" if mount_home else "--no-container-mount-home",
        "bash",
        "-euo",
        "pipefail",
        "-s",
    ]


def test_pyxis_requires_kwargs():
    with pytest.raises(ValidationError):
        PyxisRunner()


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
    workflow = Taskfile.model_validate({"version": "3", "tasks": {"default": {"runner": runner}}})
    assert isinstance(workflow.tasks["default"].runner, expected)
    assert Taskfile.model_validate_json(workflow.model_dump_json()) == workflow


@pytest.mark.parametrize(
    "runner",
    [
        SrunRunner(),
        PyxisRunner(kwargs=PyxisKwargs(container_image="model.sqsh")),
    ],
)
def test_taskfile_applies_srun_defaults_through_runner(tmp_path, runner):
    workflow = Taskfile(
        version="3",
        tasks={"default": Task(deps=["train"]), "train": Task(runner=runner)},
    )

    resolved = workflow.resolve(tmp_path)

    assert resolved.tasks["train"].runner.kwargs.job_name == "train"
    assert resolved.tasks["train"].runner.kwargs.output == str(tmp_path.resolve() / "%x.log")


def test_taskfile_preserves_explicit_srun_defaults(tmp_path):
    runner = SrunRunner(kwargs=SrunKwargs(job_name="custom", output="custom.log"))
    workflow = Taskfile(
        version="3",
        tasks={"default": Task(deps=["train"]), "train": Task(runner=runner)},
    )

    resolved = workflow.resolve(tmp_path)

    assert resolved.tasks["train"].runner.kwargs.job_name == "custom"
    assert resolved.tasks["train"].runner.kwargs.output == "custom.log"


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
