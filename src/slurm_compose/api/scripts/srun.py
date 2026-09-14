from dataclasses import dataclass, field
from pathlib import Path

from slurm_compose.config import SRUN_ERROR, SRUN_OUTPUT

from .base import Script
from .utils import fields_to_argv, maybe_update_fields, resolve_log_template


@dataclass
class SrunScript(Script):
    """Srun step script.

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

    kill_on_bad_exit: bool = field(default=True)

    exact: bool = field(default=True)

    overlap: bool = field(default=False)

    extra_argv: list[str] = field(default_factory=list, metadata={"argv": False})

    def __post_init__(self):
        if not self.job_name:
            raise ValueError("job_name cannot be empty.")

        super().__post_init__()

        self.pre_argv()

    def maybe_update(self, force: bool = False, **kwargs):
        """Updates attributes that are non-null.

        When force is True, then all non-null kwargs are set.
        """
        maybe_update_fields(self, force=force, **kwargs)
        self.pre_argv()

    def pre_argv(self):
        self.output = resolve_log_template(self.output, SRUN_OUTPUT)
        self.error = resolve_log_template(self.error, SRUN_ERROR)

        if not self.kill_on_bad_exit:
            self.kill_on_bad_exit = None

        if not self.exact:
            self.exact = None

        if not self.overlap:
            self.overlap = None

        if isinstance(self.gpus_per_node, int) and self.gpus_per_node < 0:
            self.gpus_per_node = None

    @property
    def argv(self) -> list[str]:
        self.pre_argv()

        srun_argv = fields_to_argv(self)

        return [str(arg) for arg in ["srun"] + srun_argv + self.extra_argv + ["\\\n"] + self.command]
