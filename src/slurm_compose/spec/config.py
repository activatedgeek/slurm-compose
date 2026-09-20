import hashlib
import json
import shlex
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from pkgutil import resolve_name
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator
from ruamel.yaml import YAML

from slurm_compose.config import PROJECT_NAME, STATE_HOME

from .base import BaseConfig, StepKwargsConfig
from .resolve import resolve_vars
from .slurm import PyxisConfig, SbatchConfig, SrunConfig


class DependencyConfig(BaseConfig):
    """A startup gate, not an instruction to restart downstream steps."""

    condition: Literal["healthy", "completed_successfully"]


class StepConfig(BaseConfig):
    """A named shell/srun/Pyxis command with orchestration settings."""

    type: str | None = None
    kwargs: PyxisConfig | SrunConfig | StepKwargsConfig | None = None
    command: str | list[str]
    before_script: list[str] = Field(default_factory=list)
    after_script: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    depends_on: dict[str, DependencyConfig] = Field(default_factory=dict)

    @property
    def scriptv(self) -> list[str]:
        """Return before-script, command, and after-script entries."""
        if self.kwargs is not None:
            script = self.kwargs.make_script(self.command)
        elif isinstance(self.command, list):
            script = shlex.join(self.command)
        else:
            script = self.command
        return [*self.before_script, script, *self.after_script]

    @model_validator(mode="before")
    @classmethod
    def resolve_custom_kwargs(cls, value: object) -> object:
        if not isinstance(value, dict) or value.get("kwargs") is None:
            return value
        if value.get("type") is None:
            kwargs_class = PyxisConfig if "container_image" in value["kwargs"] else SrunConfig
            return {**value, "kwargs": kwargs_class.model_validate(value["kwargs"])}
        try:
            step_class = resolve_name(value["type"])
        except (ImportError, AttributeError, ValueError) as exc:
            raise ValueError(f"unable to resolve custom step type {value['type']!r}") from exc
        if not isinstance(step_class, type) or not issubclass(step_class, StepKwargsConfig):
            raise ValueError(f"custom type {value['type']!r} must derive from StepKwargsConfig")
        try:
            kwargs = step_class.model_validate(value["kwargs"])
        except AttributeError as exc:
            raise ValueError(f"custom type {value['type']!r} must derive from StepKwargsConfig") from exc
        return {**value, "kwargs": kwargs}

    @field_validator("command")
    @classmethod
    def validate_command(cls, command: str | list[str]) -> str | list[str]:
        if isinstance(command, str):
            if not command.strip() or "\x00" in command:
                raise ValueError("command must be a nonempty string without NUL characters")
            return command
        if not command or any(not item or "\x00" in item for item in command):
            raise ValueError("exec-form command requires nonempty strings without NUL characters")
        return command

    @field_validator("before_script", "after_script")
    @classmethod
    def validate_scripts(cls, commands: list[str]) -> list[str]:
        if any(not command.strip() or "\x00" in command for command in commands):
            raise ValueError("script commands must be nonempty strings without NUL characters")
        return commands

    @model_validator(mode="after")
    def validate_type(self) -> Self:
        if self.type is not None:
            try:
                step_class = resolve_name(self.type)
            except (ImportError, AttributeError, ValueError) as exc:
                raise ValueError(f"unable to resolve custom type {self.type!r}") from exc
            if not isinstance(step_class, type) or not issubclass(step_class, StepKwargsConfig):
                raise ValueError(f"custom type {self.type!r} must derive from StepKwargsConfig")
            if self.kwargs is None or not isinstance(self.kwargs, step_class):
                raise ValueError(f"custom type {self.type!r} does not match kwargs")
        return self


class ComposeConfig(BaseConfig):
    """One document, one sbatch allocation, and a nonempty set of named steps."""

    schema_version: Literal[1] = 1
    vars: dict[str, str | int | float | bool] = Field(default_factory=dict)
    name: str | None = Field(default=None, validate_default=True)
    directory: str | Path | None = Field(default=None, validate_default=True)
    sbatch: SbatchConfig | None = None
    steps: Annotated[dict[str, StepConfig], Field(min_length=1)]

    @property
    def hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_yaml(
        cls,
        file: str | Path,
        job_name: str | None = None,
        job_dir: str | Path | None = None,
        runtime_vars: dict[str, object] | None = None,
    ) -> Self:
        yaml = YAML(typ="safe")
        yaml.allow_duplicate_keys = False
        data = yaml.load(Path(file).read_text())
        if not isinstance(data, dict):
            raise ValueError("Compose configuration must contain a top-level mapping")
        data = dict(data)
        if job_name is not None:
            data["name"] = job_name
        if job_dir is not None:
            data["directory"] = job_dir
        if runtime_vars:
            data["vars"] = {**data.get("vars", {}), **runtime_vars}
        return cls.model_validate(data)

    @model_validator(mode="before")
    @classmethod
    def resolve_variables(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        variables = value.get("vars", {})
        if not isinstance(variables, dict):
            return value
        return resolve_vars(value, variables)

    @field_validator("name", mode="after")
    @classmethod
    def default_name(cls, name: str | None) -> str:
        if name is not None:
            return name
        if PROJECT_NAME is not None:
            return PROJECT_NAME
        raise ValueError("name is required when SCOMPOSE_PROJECT_NAME is unset")

    @field_validator("directory", mode="after")
    @classmethod
    def default_directory(cls, directory: str | Path | None) -> Path:
        return STATE_HOME if directory is None else Path(directory)

    @model_validator(mode="after")
    def validate_step_graph(self) -> Self:
        if self.sbatch is not None and self.sbatch.job_name is None:
            self.sbatch = self.sbatch.model_copy(update={"job_name": self.name})

        for name, step in self.steps.items():
            for dependency_name in step.depends_on:
                if dependency_name not in self.steps:
                    raise ValueError(f"step {name!r} depends on unknown step {dependency_name!r}")

        try:
            TopologicalSorter({name: step.depends_on for name, step in self.steps.items()}).prepare()
        except CycleError as exc:
            raise ValueError(f"step dependencies contain a cycle: {' -> '.join(exc.args[1])}") from exc

        if (
            self.sbatch is not None
            and self.sbatch.completion.mode == "step"
            and self.sbatch.completion.step not in self.steps
        ):
            raise ValueError(f"completion references unknown step {self.sbatch.completion.step!r}")
        return self
