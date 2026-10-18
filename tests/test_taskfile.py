from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from slurm_compose.taskfile import Taskfile


def test_debug_taskfile():
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    data = YAML(typ="safe").load(path.read_text())
    workflow = Taskfile.model_validate(data)

    assert workflow.tasks["debug"].deps == ["prepare"]
    assert len(workflow.tasks["debug"].cmds) == 2


def test_to_yaml_round_trip():
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    workflow = Taskfile.model_validate(YAML(typ="safe").load(path.read_text()))
    # Include templates and mutations to initially default-valued mappings.
    workflow.tasks["debug"].env["MODEL"] = "{{.MODEL}}"

    data = YAML(typ="safe").load(workflow.to_yaml())

    assert Taskfile.model_validate(data) == workflow
    assert data["version"] == "3"
    assert "dir" not in data["tasks"]["debug"]
    assert "vars" not in data
    assert data["tasks"]["default"] == {
        "desc": "Run the simple debug workflow",
        "deps": ["debug"],
    }


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
