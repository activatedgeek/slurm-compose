"""Runner specifications and command transformation interfaces."""

from typing import Annotated, Literal

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

    type: Literal["srun"] = "srun"
    kwargs: SrunKwargs = Field(default_factory=SrunKwargs)


class PyxisKwargs(SrunKwargs):
    """Slurm step arguments extended with Pyxis container settings."""

    container_image: NonEmptyString
    container_mounts: list[NonEmptyString] = Field(default_factory=list)
    container_workdir: NonEmptyString | None = None
    container_mount_home: bool = False


class PyxisRunner(Runner):
    """Launch a containerized Slurm job step through Pyxis."""

    type: Literal["pyxis"] = "pyxis"
    kwargs: PyxisKwargs


RunnerConfig = Annotated[BashRunner | SrunRunner | PyxisRunner, Field(discriminator="type")]
