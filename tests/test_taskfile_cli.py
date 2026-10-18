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


@pytest.mark.parametrize("flag", ["-o", "--output-dir"])
def test_output_dir_argument(flag, tmp_path):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    command = tyro.cli(TaskCommand, args=["-t", str(path), flag, str(tmp_path)])
    assert command.output_dir == tmp_path


@pytest.mark.parametrize("flags", [[], ["-f"], ["--force"]])
def test_overwrite_generated_taskfile(monkeypatch, tmp_path, flags):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    generated = tmp_path / "Taskfile.yml"
    generated.write_text("existing content\n")
    command = tyro.cli(TaskCommand, args=["-t", str(path), "-o", str(tmp_path), *flags])
    started = []

    @contextmanager
    def popen(args, **kwargs):
        started.append(True)
        assert generated.read_text() == command.config.to_yaml()
        yield SimpleNamespace(stdout=StringIO(), stderr=StringIO(), wait=lambda: 0)

    monkeypatch.setattr("slurm_compose.taskfile.task.subprocess.Popen", popen)

    if flags:
        assert command.force is True
        assert command() == 0
        assert started == [True]
    else:
        assert command.force is False
        with pytest.raises(FileExistsError):
            command()
        assert generated.read_text() == "existing content\n"
        assert started == []


def test_print_outputs_yaml_without_execution(monkeypatch, capsys):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    command = tyro.cli(TaskCommand, args=["-t", str(path), "--print"])

    def unexpected_call(*args, **kwargs):
        pytest.fail("print must not create temporary files or execute Task")

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
@pytest.mark.parametrize("persistent", [True, False])
def test_execute_generated_taskfile(monkeypatch, capsys, tmp_path, returncode, silent, raw, persistent):
    path = Path(__file__).parent / "configs" / "Taskfile.yml"
    output_dir = tmp_path / "nested" / "output" if persistent else None
    command = TaskCommand(taskfile=path, silent=silent, raw=raw, output_dir=output_dir)
    generated_paths = []

    @contextmanager
    def popen(args, **kwargs):
        prefix = ["task", *(["--silent"] if silent else []), "--exit-code", "--dir", str(path.resolve().parent)]
        assert args[:-2] == prefix
        assert args[-2] == "--taskfile"
        assert kwargs == {"stdout": PIPE, "stderr": PIPE, "text": True, "errors": "replace"}
        generated = Path(args[-1])
        generated_paths.append(generated)
        assert generated != path.resolve()
        data = YAML(typ="safe").load(generated.read_text())
        assert data["env"]["TASK_HASH"] == command.config.hash
        for name, task in data["tasks"].items():
            assert task["env"]["TASK_NAME"] == name
            assert "runner" not in task
        exclude = {"tasks": {"__all__": {"runner"}}}
        assert Taskfile.model_validate(data).model_dump(exclude=exclude) == command.config.resolve().model_dump(
            exclude=exclude
        )
        yield SimpleNamespace(
            stdout=StringIO("[bold]literal stdout[/bold]\nfinal line"),
            stderr=StringIO("literal stderr\n"),
            wait=lambda: returncode,
        )

    monkeypatch.setattr("slurm_compose.taskfile.task.subprocess.Popen", popen)

    assert command() == returncode
    if persistent:
        assert generated_paths[0] == output_dir / "Taskfile.yml"
        assert generated_paths[0].is_file()
    else:
        assert not generated_paths[0].parent.exists()
    output = capsys.readouterr()
    if persistent:
        assert "Warning:" not in output.err
    else:
        assert "Warning: Using temporary output directory" in output.err
        assert str(generated_paths[0].parent) in output.err
        assert "removed after execution" in output.err
    assert "literal stdout" in output.out
    assert "final line" in output.out
    if raw:
        assert output.out == "[bold]literal stdout[/bold]\nfinal line"
        assert output.err.endswith("literal stderr\n")
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
