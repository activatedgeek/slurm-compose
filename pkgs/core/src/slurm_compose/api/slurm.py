from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader

from .base import BaseArgs
from .utils import fields_to_argv


@dataclass
class SlurmJobStep(BaseArgs):
    """Slurm job step.

    Sets up the command and environment to run in sbatch file.
    """

    command: str | list[str] = field(default_factory=list)

    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.command, str):
            self.command = self.command.split()

        if not self.command:
            raise ValueError("command cannot be empty.")

    @property
    def argv(self) -> list[str]:
        return [str(arg) for arg in self.command]


@dataclass
class SrunJobStep(SlurmJobStep):
    """Srun step arguments.

    Each step is prefixed with srun and appropriate args.
    """

    job_name: str | None = field(default=None)

    nodes: int | None = field(default=None)

    ntasks_per_node: int | None = field(default=None)

    cpus_per_task: int | None = field(default=None)

    gpus_per_node: int | None = field(default=None)

    mem: str | None = field(default=None)

    output: str | Path | None = field(default=None)

    error: str | Path | None = field(default=None)

    wait: int = field(default=10)

    kill_on_bad_exit: int = field(default=1)

    extra_argv: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.job_name:
            raise ValueError("job_name cannot be empty.")

        if not self.error:
            self.error = self.output

        super().__post_init__()

    @property
    def argv(self) -> list[str]:
        srun_argv = fields_to_argv(self, ignore_keys=SlurmJobStep.fields().keys() | {"extra_argv"})

        return [str(arg) for arg in ["srun"] + srun_argv + self.extra_argv + self.command]


@dataclass
class SlurmJob(BaseArgs):
    """Slurm Job Arguments

    All these arguments are passed to sbatch. See https://slurm.schedmd.com/sbatch.html for docs.

    `extras` is a catch all for arguments that are currently part of the typed dataclass.
    """

    job_name: str | None = field(default=None)

    account: str | None = field(default=None)

    partition: str | None = field(default=None)

    qos: str | None = field(default=None)

    time: str | timedelta | None = field(default=None)

    nodes: int = field(default=1)

    ntasks_per_node: int = field(default=8)

    cpus_per_task: int | None = field(default=None)

    gpus_per_node: int | None = field(default=None)

    mem: str | None = field(default=None)

    output: str | Path | None = field(default=None)

    error: str | Path | None = field(default=None)

    open_mode: Literal["append", "truncate"] = field(default="append")

    extra_argv: list[str] = field(default_factory=list)

    steps: list[SlurmJobStep] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.time, timedelta):
            total_seconds = int(self.time.total_seconds())
            days, remainder = divmod(total_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            if days > 0:
                self.time = f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                self.time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        if not self.error:
            self.error = self.output

    def materialize(self, template: str = "slurm.sh.j2", template_dir: str | list[str] | None = None) -> str:
        if isinstance(template_dir, str):
            template_dir = [template_dir]
        template_dir = template_dir or []

        env = Environment(
            loader=FileSystemLoader(template_dir + [Path(__file__).parent / "templates"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        template = env.get_template(template)

        sbatch_argv = fields_to_argv(
            self, ignore_keys=BaseArgs.fields().keys() | {"extra_argv", "steps"}, equals_separated=True
        )

        return template.render(
            sbatch_argv=sbatch_argv + (self.extra_argv or []),
            steps=self.steps,
        )
