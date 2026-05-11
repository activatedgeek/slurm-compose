from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from slurm_compose.api.base import BaseArgs


@dataclass
class SlurmJobStep(BaseArgs):
    command: list[str] = field(default_factory=list)

    env: dict[str, str] = field(default_factory=dict)


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

    time: str | None = field(default=None)

    nodes: int = field(default=1)

    ntasks_per_node: int = field(default=8)

    cpus_per_task: int = field(default=1)

    gpus_per_node: int | None = field(default=None)

    mem: str | None = field(default=None)

    output: str | Path | None = field(default=None)

    error: str | Path | None = field(default=None)

    extras: list[str] = field(default_factory=list)

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

        self.env = Environment(
            loader=FileSystemLoader([Path(__file__).parent / "templates"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def materialize(self, template: str = "slurm.sh.j2") -> str:
        template = self.env.get_template(template)
        return template.render(**self.to_dict())
