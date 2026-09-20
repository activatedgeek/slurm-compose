import shlex
from datetime import timedelta
from pathlib import Path
from typing import ClassVar, Literal, Self
from warnings import warn

from pydantic import Field, field_validator, model_validator
from pytimeparse import parse as timeparse

from .base import BaseConfig, StepKwargsConfig


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

    def make_script(self, command: str | list[str]) -> str:
        prefix = shlex.join(["srun", *self.argv])
        return prefix + " \\\n" + super().make_script(command)


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
