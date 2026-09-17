import hashlib
import json
from datetime import timedelta
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from pkgutil import resolve_name
from typing import Annotated, ClassVar, Literal, Self
from warnings import warn

from pydantic import Field, field_validator, model_validator
from pytimeparse import parse as timeparse
from ruamel.yaml import YAML

from slurm_compose.config import PROJECT_NAME, STATE_HOME

from .base import BaseConfig
from .resolve import resolve_vars


class CompletionConfig(BaseConfig):
    """Determine when the workflow finishes: all steps, any step, or one step."""

    mode: Literal["all", "any", "step"] = "any"
    step: str | None = None

    @model_validator(mode="after")
    def validate_step(self) -> Self:
        if self.mode == "step" and self.step is None:
            raise ValueError("completion step is required when mode is 'step'")
        if self.mode != "step" and self.step is not None:
            raise ValueError("completion step is only valid when mode is 'step'")
        return self


class StepKwargsConfig(BaseConfig):
    """Base configuration for command launcher arguments."""

    _argv_exclude: ClassVar[set[str]] = set()

    @property
    def argv(self) -> list[str]:
        """Return launcher options as equals-separated command arguments."""
        args: list[str] = []
        for name in type(self).model_fields:
            if name in type(self)._argv_exclude:
                continue
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            option = f"--{name.replace('_', '-')}"
            if isinstance(value, bool):
                args.append(option if value else f"--no-{name.replace('_', '-')}")
            elif isinstance(value, list):
                args.append(f"{option}={','.join(str(item) for item in value)}")
            else:
                args.append(f"{option}={value}")
        return args


class SlurmResourceConfig(StepKwargsConfig):
    """Explicit resource flags shared by sbatch and srun."""

    _argv_exclude: ClassVar[set[str]] = {"extra_argv"}

    nodes: int = Field(gt=0)
    ntasks_per_node: int = Field(gt=0)
    cpus_per_task: int = Field(gt=0)
    gpus_per_node: int = Field(ge=0)
    mem: str = Field(default=None)

    @property
    def argv(self) -> list[str]:
        return super().argv + self.extra_argv


class SbatchConfig(SlurmResourceConfig):
    """Explicit resource flags for sbatch."""

    _argv_exclude: ClassVar[set[str]] = SlurmResourceConfig._argv_exclude | {"completion"}

    job_name: str | None = None
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    time: str | timedelta | None = None
    begin: str | None = None
    output: str | Path | None = None
    error: str | Path | None = None
    open_mode: Literal["append", "truncate"] = "append"
    requeue: Literal[True] | None = True
    array: str | None = None
    extra_argv: list[str] = Field(default_factory=list, description="Additional equals-separated sbatch arguments.")
    completion: CompletionConfig = Field(default_factory=CompletionConfig)

    @field_validator("time", mode="after")
    @classmethod
    def parse_time(cls, time: str | timedelta | None) -> str | None:
        if time is None:
            return time
        if isinstance(time, str):
            parsed_seconds = timeparse(time, granularity="seconds")
            if parsed_seconds is None:
                warn(f"could not parse sbatch time {time!r}; leaving it unchanged", RuntimeWarning, stacklevel=2)
                return time
            time = timedelta(seconds=parsed_seconds)

        total_seconds = int(time.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        if days > 0:
            return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class SrunConfig(SlurmResourceConfig):
    """The launch arguments of api.scripts.srun.SrunScript."""

    job_name: str | None = None
    output: str | Path | None = None
    error: str | Path | None = None
    wait: int = Field(default=10, ge=0)
    kill_on_bad_exit: Literal[0, 1] = 1
    exact: Literal[True] | None = True
    overlap: Literal[True] | None = None
    extra_argv: list[str] = Field(default_factory=list, description="Additional equals-separated srun arguments.")


class PyxisConfig(SrunConfig):
    """srun arguments extended with api.scripts.pyxis.PyxisScript options."""

    container_image: str | Path
    container_mounts: str | list[str] = Field(default_factory=list)
    container_workdir: str | Path | None = None
    container_mount_home: bool = False

    @field_validator("container_image", "container_workdir", mode="after")
    @classmethod
    def normalize_paths(cls, path: str | Path | None) -> Path | None:
        return None if path is None else Path(path)


class DependencyConfig(BaseConfig):
    """A startup gate, not an instruction to restart downstream steps."""

    condition: Literal["started", "healthy", "completed_successfully"]


class HealthcheckConfig(BaseConfig):
    """Compose-style health check; a test is required unless explicitly disabled.

    String tests are shell commands. Lists use CMD (argv), CMD-SHELL (one shell
    command), or NONE (disabled). Execution belongs to the future supervisor.
    Duration values are strings such as ``500ms``, ``30s``, or ``1m30s``.
    """

    test: str | list[str] | None = None
    interval: str = "30s"
    timeout: str = "30s"
    retries: int = Field(default=3, gt=0)
    start_period: str = "0s"
    start_interval: str = "5s"
    disable: bool = False

    @field_validator("test")
    @classmethod
    def validate_test(cls, test: str | list[str] | None) -> str | list[str] | None:
        if test is None:
            return test
        if isinstance(test, str):
            if not test.strip() or "\x00" in test:
                raise ValueError("healthcheck test must be a nonempty shell command without NUL characters")
            return test
        if not test or any("\x00" in arg for arg in test):
            raise ValueError("healthcheck test requires CMD, CMD-SHELL, or NONE")
        if test[0] == "NONE" and len(test) == 1:
            return test
        if test[0] == "CMD" and len(test) >= 2 and test[1]:
            return test
        if test[0] == "CMD-SHELL" and len(test) == 2 and test[1].strip():
            return test
        raise ValueError("healthcheck test must be [CMD, executable, ...], [CMD-SHELL, command], or [NONE]")

    @model_validator(mode="after")
    def require_test(self) -> Self:
        if not self.disable and self.test is None:
            raise ValueError("an enabled healthcheck requires a test")
        return self

    @property
    def enabled(self) -> bool:
        return not self.disable and self.test is not None and self.test != ["NONE"]


class StepConfig(BaseConfig):
    """A named shell/srun/Pyxis command with orchestration settings.

    A missing kwargs block denotes a shell step. The optional type field resolves
    any custom BaseConfig model by its fully qualified name.
    Commands use Docker Compose's shell string or exec list forms.
    Shell steps run in the batch environment; srun/Pyxis steps run their command
    inside the launched step.
    User scripts are responsible for multinode coordination.
    """

    type: str | None = None
    kwargs: PyxisConfig | SrunConfig | StepKwargsConfig | None = None
    command: str | list[str]
    before_script: list[str] = Field(default_factory=list)
    after_script: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    depends_on: dict[str, DependencyConfig] = Field(default_factory=dict)
    healthcheck: HealthcheckConfig | None = None

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
                srun_class = resolve_name(self.type)
            except (ImportError, AttributeError, ValueError) as exc:
                raise ValueError(f"unable to resolve custom type {self.type!r}") from exc
            if not isinstance(srun_class, type) or not issubclass(srun_class, StepKwargsConfig):
                raise ValueError(f"custom type {self.type!r} must derive from StepKwargsConfig")
            if self.kwargs is None or not isinstance(self.kwargs, srun_class):
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
    topology: dict[str, dict] = Field(default_factory=dict, frozen=True, exclude=True)

    @property
    def hash(self) -> str:
        """Return a stable digest of the fully resolved configuration."""
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
        """Load, override, resolve, and validate a Compose configuration file."""
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
        """Resolve vars references across the raw configuration before validation."""
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
            for dependency_name, dependency in step.depends_on.items():
                if dependency_name not in self.steps:
                    raise ValueError(f"step {name!r} depends on unknown step {dependency_name!r}")
                target = self.steps[dependency_name]
                if dependency.condition == "healthy" and (
                    target.healthcheck is None or not target.healthcheck.enabled
                ):
                    raise ValueError(f"step {dependency_name!r} needs an enabled healthcheck for a healthy dependency")

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
        children = {name: {} for name in self.steps}
        for name, step in self.steps.items():
            for dependency_name in step.depends_on:
                children[dependency_name][name] = {}

        def expand(name: str) -> dict[str, dict]:
            return {child: expand(child) for child in children[name]}

        topology = {name: expand(name) for name, step in self.steps.items() if not step.depends_on}
        return self.model_copy(update={"topology": topology})
