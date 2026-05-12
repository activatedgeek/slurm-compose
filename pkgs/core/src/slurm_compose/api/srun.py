from dataclasses import dataclass, field
from pathlib import Path

from .slurm import SlurmJobStep
from .utils import fields_to_argv


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

    exact: bool = field(default=True)

    overlap: bool = field(default=False)

    extra_argv: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.job_name:
            raise ValueError("job_name cannot be empty.")

        if not self.exact:
            self.exact = None

        if not self.overlap:
            self.overlap = None

        if not self.error:
            self.error = self.output

        super().__post_init__()

    @property
    def argv(self) -> list[str]:
        srun_argv = fields_to_argv(self, ignore_keys=SlurmJobStep.fields().keys() | {"extra_argv"})

        return [str(arg) for arg in ["srun"] + srun_argv + self.extra_argv + self.command]
