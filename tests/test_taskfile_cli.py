from pathlib import Path
from subprocess import CompletedProcess

import pytest
import tyro
from ruamel.yaml import YAML
from ruamel.yaml.constructor import DuplicateKeyError

from slurm_compose.taskfile import Taskfile
from slurm_compose.taskfile.cli import TaskCommand


@pytest.mark.parametrize("flag", ["-t", "--taskfile"])
def test_load_taskfile(flag):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    command = tyro.cli(TaskCommand, args=[flag, str(path)])

    assert command.taskfile == path
    assert isinstance(command.config, Taskfile)
    assert command.config.tasks["debug"].deps == ["prepare"]
    assert len(command.config.tasks["debug"].cmds) == 2


def test_reject_duplicate_yaml_keys(tmp_path):
    path = tmp_path / "Taskfile.yml"
    path.write_text("version: '3'\nversion: '3'\ntasks: {}\n")

    with pytest.raises(DuplicateKeyError):
        TaskCommand(taskfile=path)


@pytest.mark.parametrize("returncode", [0, 7])
def test_execute_generated_taskfile(monkeypatch, returncode):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    command = TaskCommand(taskfile=path)
    generated_paths = []

    def run(args, *, check):
        assert args[:4] == ["task", "--exit-code", "--dir", str(path.resolve().parent)]
        assert args[4] == "--taskfile"
        assert check is False
        generated = Path(args[5])
        generated_paths.append(generated)
        assert generated != path.resolve()
        assert Taskfile.model_validate(YAML(typ="safe").load(generated.read_text())) == command.config
        return CompletedProcess(args, returncode)

    monkeypatch.setattr("slurm_compose.taskfile.cli.subprocess.run", run)

    assert command() == returncode
    assert not generated_paths[0].parent.exists()
