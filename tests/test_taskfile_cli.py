from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from subprocess import PIPE
from types import SimpleNamespace

import pytest
import tyro
from ruamel.yaml import YAML
from ruamel.yaml.constructor import DuplicateKeyError

from slurm_compose.taskfile import Taskfile
from slurm_compose.taskfile.cli import TaskCommand, main


@pytest.mark.parametrize("flag", ["-t", "--taskfile"])
def test_load_taskfile(flag):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    command = tyro.cli(TaskCommand, args=[flag, str(path)])

    assert command.taskfile == path
    assert isinstance(command.config, Taskfile)
    assert command.config.tasks["debug"].deps == ["prepare"]
    assert len(command.config.tasks["debug"].cmds) == 3


def test_reject_duplicate_yaml_keys(tmp_path):
    path = tmp_path / "Taskfile.yml"
    path.write_text("version: '3'\nversion: '3'\ntasks: {}\n")

    with pytest.raises(DuplicateKeyError):
        TaskCommand(taskfile=path)


@pytest.mark.parametrize(
    ("flags", "silent"),
    [([], True), (["-s"], True), (["--silent"], True), (["--no-silent"], False)],
)
def test_silent_argument(flags, silent):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    command = tyro.cli(TaskCommand, args=["-t", str(path), *flags])
    assert command.silent is silent


@pytest.mark.parametrize("flag", ["-r", "--raw"])
def test_raw_argument(flag):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    command = tyro.cli(TaskCommand, args=["-t", str(path), flag])
    assert command.raw is True


def test_dump_prints_yaml_without_execution(monkeypatch, capsys):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    command = tyro.cli(TaskCommand, args=["-t", str(path), "--dump"])

    def unexpected_call(*args, **kwargs):
        pytest.fail("dump must not create temporary files or execute Task")

    monkeypatch.setattr("slurm_compose.taskfile.task.subprocess.Popen", unexpected_call)
    monkeypatch.setattr("slurm_compose.taskfile.cli.TemporaryDirectory", unexpected_call)

    assert command() == 0
    output = capsys.readouterr()
    assert output.out == command.config.to_yaml()
    assert output.err == ""
    assert YAML(typ="safe").load(output.out)["env"]["TASK_HASH"] == command.config.hash


@pytest.mark.parametrize("returncode", [0, 7])
@pytest.mark.parametrize("silent", [True, False])
@pytest.mark.parametrize("raw", [True, False])
def test_execute_generated_taskfile(monkeypatch, capsys, returncode, silent, raw):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    command = TaskCommand(taskfile=path, silent=silent, raw=raw)
    generated_paths = []

    @contextmanager
    def popen(args, **kwargs):
        prefix = ["task", *(["--silent"] if silent else []), "--exit-code", "--dir", str(path.resolve().parent)]
        assert args[:-2] == prefix
        assert args[-2] == "--taskfile"
        assert kwargs == {"stdout": PIPE, "stderr": PIPE, "text": True, "errors": "replace", "bufsize": 1}
        generated = Path(args[-1])
        generated_paths.append(generated)
        assert generated != path.resolve()
        data = YAML(typ="safe").load(generated.read_text())
        assert data["env"].pop("TASK_HASH") == command.config.hash
        for name, task in data["tasks"].items():
            assert task["env"].pop("TASK_NAME") == name
        assert Taskfile.model_validate(data) == command.config
        yield SimpleNamespace(
            stdout=StringIO("[bold]literal stdout[/bold]\nfinal line"),
            stderr=StringIO("literal stderr\n"),
            wait=lambda: returncode,
        )

    monkeypatch.setattr("slurm_compose.taskfile.task.subprocess.Popen", popen)

    assert command() == returncode
    assert not generated_paths[0].parent.exists()
    output = capsys.readouterr()
    assert "literal stdout" in output.out
    assert "final line" in output.out
    if raw:
        assert output.out == "[bold]literal stdout[/bold]\nfinal line"
        assert output.err == "literal stderr\n"
    else:
        assert "[bold]" not in output.out
    assert "literal stderr" in output.err


def test_cli_reports_errors_on_stderr(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["stask", "-t", str(tmp_path / "missing.yml")])
    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err
