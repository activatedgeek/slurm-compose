"""Runner specifications and command transformation interfaces."""

import shlex
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class Runner(BaseModel):
    """Base specification for a command runner."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: str

    def __call__(self, cmds: list[str]) -> list[str]:
        """Transform a list of shell commands for this runner."""
        raise NotImplementedError


class BashRunner(Runner):
    """Execute shell commands with Bash."""

    type: Literal["bash"] = "bash"

    def __call__(self, cmds: list[str]) -> list[str]:
        """Return shell commands unchanged."""
        return cmds


class SrunKwargs(BaseModel):
    """Optional Slurm step arguments; omitted values use Slurm defaults."""

    model_config = ConfigDict(extra="forbid", strict=True)

    nodes: int | None = Field(default=None, gt=0)
    ntasks_per_node: int | None = Field(default=None, gt=0)
    cpus_per_task: int | None = Field(default=None, gt=0)
    gpus_per_node: int | None = Field(default=None, ge=0)
    mem: NonEmptyString | None = None
    extra_argv: list[NonEmptyString] = Field(default_factory=list)


class SrunRunner(Runner):
    """Launch a command as a Slurm job step."""

    delimiter: ClassVar[str] = "TASK_SRUN"
    type: Literal["srun"] = "srun"
    kwargs: SrunKwargs = Field(default_factory=SrunKwargs)

    @property
    def pre_cmd(self) -> list[str]:
        """Build the launcher and Bash prefix as individual arguments."""
        options = []
        for name, value in self.kwargs.model_dump(exclude_none=True, exclude={"extra_argv"}).items():
            option = name.replace("_", "-")
            if isinstance(value, bool):
                options.append(f"--{option}" if value else f"--no-{option}")
            elif isinstance(value, list):
                if value:
                    options.append(f"--{option}={','.join(value)}")
            else:
                options.append(f"--{option}={value}")
        return ["srun", *options, *self.kwargs.extra_argv, "bash", "-euo", "pipefail", "-s"]

    def __call__(self, cmds: list[str]) -> list[str]:
        """Wrap each complete shell command in a separate synchronous Slurm step."""
        transformed = []
        for cmd in cmds:
            delimiter = self.delimiter
            suffix = 0
            lines = set(cmd.splitlines())
            while delimiter in lines:
                suffix += 1
                delimiter = f"{self.delimiter}_{suffix}"
            body = cmd if cmd.endswith("\n") else cmd + "\n"
            transformed.append(f"{shlex.join(self.pre_cmd)} <<'{delimiter}'\n{body}{delimiter}\n")
        return transformed


class PyxisKwargs(SrunKwargs):
    """Slurm step arguments extended with Pyxis container settings."""

    container_image: NonEmptyString
    container_mounts: list[NonEmptyString] = Field(default_factory=list)
    container_workdir: NonEmptyString | None = None
    container_mount_home: bool = False


class PyxisRunner(SrunRunner):
    """Launch a containerized Slurm job step through Pyxis."""

    delimiter: ClassVar[str] = "TASK_PYXIS"
    type: Literal["pyxis"] = "pyxis"
    kwargs: PyxisKwargs = Field(...)


RunnerConfig = Annotated[BashRunner | SrunRunner | PyxisRunner, Field(discriminator="type")]
