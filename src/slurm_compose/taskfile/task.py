"""A deliberately limited subset of Taskfile schema version 3."""

import hashlib
import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from graphlib import CycleError, TopologicalSorter
from io import StringIO
from pathlib import Path
from threading import Thread
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from ruamel.yaml import YAML

from .logging import console, output_console, stream_output
from .runner import BashRunner, RunnerConfig

NonEmptyString = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Scalar = str | int | float | bool


class TaskfileSpecModel(BaseModel):
    """Reject unsupported fields and implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Task(TaskfileSpecModel):
    """Sequential shell commands, started after all dependencies succeed.

    Commands are expected to terminate. Their runtime behavior is not validated.
    A command list represents separate shell commands, not an argv array.
    """

    desc: str | None = None
    cmds: list[NonEmptyString] = Field(default_factory=list)
    deps: list[NonEmptyString] = Field(default_factory=list)
    sources: list[NonEmptyString] = Field(default_factory=list)
    generates: list[NonEmptyString] = Field(default_factory=list)
    status: list[NonEmptyString] = Field(default_factory=list)
    vars: dict[str, Scalar] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    dir: NonEmptyString | None = None
    runner: RunnerConfig = Field(default_factory=BashRunner)


class Taskfile(TaskfileSpecModel):
    """Named finite tasks; dependency-only grouping tasks are allowed.

    Variables and template expressions are preserved without evaluation.
    """

    version: Literal["3"]
    tasks: Annotated[dict[NonEmptyString, Task], Field(min_length=1)]
    vars: dict[str, Scalar] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)

    @property
    def hash(self) -> str:
        """Stable SHA-256 of configuration, excluding generated metadata."""
        data = self.model_dump(
            mode="json",
            exclude={"env": {"TASK_HASH", "TASK_DIR"}, "tasks": {"__all__": {"env": {"TASK_NAME"}}}},
        )
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve(self, output_dir: str | Path = ".") -> Self:
        """Return a transformed copy with generated task metadata."""
        task_dir = Path(output_dir).resolve()

        rendered = self.model_copy(deep=True)
        rendered.env["TASK_HASH"] = self.hash
        rendered.env["TASK_DIR"] = str(task_dir)
        for name, task in rendered.tasks.items():
            task.env["TASK_NAME"] = name
            task.runner.apply_defaults(name, task_dir)
            task.cmds = task.runner(task.cmds)
            task.cmds.append('date +%s%N > "$TASK_DIR/$TASK_NAME.done"')

            task.sources.extend(str(task_dir / f"{dependency}.done") for dependency in task.deps)
            task.generates.append(str(task_dir / f"{name}.done"))
            task.status.append('test -f "$TASK_DIR/$TASK_NAME.done"')

        return rendered

    def to_yaml(
        self,
        output_dir: str | Path | None = None,
        *,
        force: bool = False,
    ) -> str:
        """Serialize the workflow, writing ``Taskfile.yml`` when output_dir is provided."""
        render_dir = output_dir if output_dir is not None else "."
        data = self.resolve(render_dir).model_dump(
            mode="json",
            exclude_defaults=True,
            exclude={"tasks": {"__all__": {"runner"}}},
        )
        output = StringIO()
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.dump(data, output)
        taskfile_yaml = output.getvalue()

        if output_dir is not None:
            generated = Path(output_dir).resolve() / "Taskfile.yml"
            generated.parent.mkdir(parents=True, exist_ok=True)
            if generated.exists():
                if generated.read_text() != taskfile_yaml:
                    if not force:
                        raise FileExistsError(
                            f"Generated Taskfile {generated} already exists with different contents; use --force to replace it"
                        )
                    generated.write_text(taskfile_yaml)
            else:
                with generated.open("x") as output_file:
                    output_file.write(taskfile_yaml)

        return taskfile_yaml

    @contextmanager
    def proc(
        self,
        output_dir: str | Path,
        *,
        silent: bool = True,
        raw: bool = False,
        force: bool = False,
    ) -> Iterator[subprocess.Popen[str]]:
        """Write YAML in a caller-owned directory and stream task output."""

        generated = Path(output_dir).resolve() / "Taskfile.yml"
        self.to_yaml(generated.parent, force=force)

        with subprocess.Popen(
            [
                "task",
                *(["--silent"] if silent else []),
                *(["--force"] if force else []),
                "--exit-code",
                "--dir",
                str(generated.parent),
                "--taskfile",
                str(generated),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        ) as process:
            readers = [
                Thread(target=stream_output, args=(process.stdout, output_console), kwargs={"raw": raw}),
                Thread(target=stream_output, args=(process.stderr, console), kwargs={"raw": raw}),
            ]
            for reader in readers:
                reader.start()
            try:
                yield process
            finally:
                for reader in readers:
                    reader.join()
                process.wait()

    @model_validator(mode="after")
    def validate_dependencies(self) -> Self:
        if "default" not in self.tasks:
            raise ValueError("Taskfile must contain a 'default' task")
        for name, task in self.tasks.items():
            for dependency in task.deps:
                if dependency not in self.tasks:
                    raise ValueError(f"task {name!r} depends on unknown task {dependency!r}")
        try:
            TopologicalSorter({name: task.deps for name, task in self.tasks.items()}).prepare()
        except CycleError as exc:
            raise ValueError("task dependencies contain a cycle") from exc
        return self
