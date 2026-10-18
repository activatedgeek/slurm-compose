from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from slurm_compose.taskfile import BashRunner, SrunRunner, Taskfile


def test_debug_taskfile():
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    data = YAML(typ="safe").load(path.read_text())
    workflow = Taskfile.model_validate(data)

    assert workflow.tasks["debug"].deps == ["prepare"]
    assert len(workflow.tasks["debug"].cmds) == 3


def test_to_yaml_round_trip():
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    workflow = Taskfile.model_validate(YAML(typ="safe").load(path.read_text()))
    # Include templates and mutations to initially default-valued mappings.
    workflow.tasks["debug"].env["MODEL"] = "{{.MODEL}}"

    data = YAML(typ="safe").load(workflow.to_yaml())

    assert data["env"].pop("TASK_HASH") == workflow.hash
    for name, task in data["tasks"].items():
        assert task["env"].pop("TASK_NAME") == name
    assert Taskfile.model_validate(data) == workflow
    assert data["version"] == "3"
    assert "dir" not in data["tasks"]["debug"]
    assert "vars" not in data
    assert data["tasks"]["default"] == {
        "desc": "Run the simple debug workflow",
        "deps": ["debug"],
        "env": {},
    }


def test_to_yaml_transforms_commands_without_mutating_source(monkeypatch):
    workflow = Taskfile.model_validate(
        {
            "version": "3",
            "tasks": {
                "prepare": {"cmds": ["echo prepare"]},
                "run": {
                    "deps": ["prepare"],
                    "runner": {"type": "bash"},
                    "env": {"MODEL": "qwen3"},
                    "cmds": ["echo one", "echo two"],
                },
            },
        }
    )
    original = workflow.model_dump()

    def transform(self, cmds):
        cmds[:] = [f"wrapped {cmd}" for cmd in cmds]
        return cmds

    monkeypatch.setattr(BashRunner, "__call__", transform)
    model = workflow.resolve()
    assert model is not workflow
    assert model.env["TASK_HASH"] == workflow.hash
    assert model.tasks["run"].env["TASK_NAME"] == "run"
    assert model.tasks["run"].cmds == ["wrapped echo one", "wrapped echo two"]
    assert workflow.model_dump() == original
    rendered = workflow.to_yaml()
    data = YAML(typ="safe").load(rendered)

    assert data["tasks"]["prepare"]["cmds"] == ["wrapped echo prepare"]
    assert data["tasks"]["run"]["cmds"] == ["wrapped echo one", "wrapped echo two"]
    assert data["tasks"]["run"]["deps"] == ["prepare"]
    assert data["tasks"]["run"]["env"] == {"MODEL": "qwen3", "TASK_NAME": "run"}
    assert all("runner" not in task for task in data["tasks"].values())
    assert workflow.model_dump() == original
    assert workflow.to_yaml() == rendered


def test_hash_is_stable_and_ignores_generated_value():
    workflow = Taskfile.model_validate(
        {
            "version": "3",
            "env": {"MODEL": "qwen3", "DATASET": "test"},
            "tasks": {"run": {"cmds": ["echo hello"]}},
        }
    )
    reordered = Taskfile.model_validate(
        {
            "tasks": {"run": {"cmds": ["echo hello"], "runner": {"type": "bash"}}},
            "env": {"DATASET": "test", "MODEL": "qwen3", "TASK_HASH": "old-value"},
            "version": "3",
        }
    )
    assert workflow.hash == reordered.hash
    assert len(workflow.hash) == 64
    assert int(workflow.hash, 16) >= 0
    rendered = YAML(typ="safe").load(reordered.to_yaml())
    assert rendered["env"] == {"DATASET": "test", "MODEL": "qwen3", "TASK_HASH": workflow.hash}
    assert "TASK_HASH" not in workflow.env
    assert reordered.env["TASK_HASH"] == "old-value"


def test_to_yaml_overrides_task_names_without_mutating_source():
    workflow = Taskfile.model_validate(
        {
            "version": "3",
            "tasks": {
                "prepare": {"cmds": ["echo prepare"], "env": {"TASK_NAME": "old", "MODEL": "qwen3"}},
                "default": {"deps": ["prepare"]},
            },
        }
    )
    original = workflow.model_dump()
    data = YAML(typ="safe").load(workflow.to_yaml())
    assert data["tasks"]["prepare"]["env"] == {"TASK_NAME": "prepare", "MODEL": "qwen3"}
    assert data["tasks"]["default"]["env"] == {"TASK_NAME": "default"}
    assert data["env"]["TASK_HASH"] == workflow.hash
    assert workflow.model_dump() == original


@pytest.mark.parametrize("change", ["commands", "environment", "runner"])
def test_hash_changes_with_configuration(change):
    workflow = Taskfile.model_validate({"version": "3", "tasks": {"run": {"cmds": ["echo hello"]}}})
    previous = workflow.hash
    if change == "commands":
        workflow.tasks["run"].cmds.append("echo world")
    elif change == "environment":
        workflow.env["MODEL"] = "qwen3"
    else:
        workflow.tasks["run"].runner = SrunRunner()
    assert workflow.hash != previous


@pytest.mark.parametrize(
    "runner",
    [
        {"type": "srun"},
        {"type": "pyxis", "kwargs": {"container_image": "model.sqsh"}},
    ],
)
def test_to_yaml_rejects_unimplemented_runners(runner):
    workflow = Taskfile.model_validate(
        {
            "version": "3",
            "tasks": {"run": {"runner": runner, "cmds": ["echo hello"]}},
        }
    )
    with pytest.raises(NotImplementedError):
        workflow.to_yaml()


def test_finite_workflow_preserves_templates_and_explicit_fields():
    data = {
        "version": "3",
        "vars": {"MODEL": "qwen3", "NODES": 1},
        "env": {"DATA_ROOT": "/shared/data"},
        "tasks": {
            "prepare": {"cmds": ["python prepare.py"]},
            "train": {
                "deps": ["prepare"],
                "env": {"MODEL": "{{.MODEL}}"},
                "cmds": ["python train.py", "python export.py"],
            },
            "all": {"deps": ["train"]},
        },
    }
    workflow = Taskfile.model_validate(data)
    assert workflow.tasks["train"].env["MODEL"] == "{{.MODEL}}"
    assert workflow.model_dump(exclude_unset=True) == data


@pytest.mark.parametrize(
    "tasks",
    [
        {"a": {"deps": ["missing"]}},
        {"a": {"deps": ["b"]}, "b": {"deps": ["a"]}},
        {"a": {"deps": ["a"]}},
        {"a": {"cmds": [" "]}},
        {"a": {"cmds": [{"task": "b"}]}},
        {"a": {"cmds": ["echo hello"], "runner": "srun"}},
        {"a": {"env": {"NODES": 1}}},
        {"a": "echo hello"},
        {},
    ],
)
def test_rejects_invalid_or_unsupported_tasks(tasks):
    with pytest.raises(ValidationError):
        Taskfile.model_validate({"version": "3", "tasks": tasks})
