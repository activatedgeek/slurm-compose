from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from slurm_compose.api.base import BaseArgs


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
        if not self.error:
            self.error = self.output

        super().__post_init__()

    @property
    def argv(self) -> list[str]:
        def _handle_arg(k):
            arg_name = f"--{k.replace('_', '-')}"
            arg_val = getattr(self, k)
            args = []
            if isinstance(arg_val, bool):
                if arg_val:
                    args = [arg_name]
            else:
                args = [arg_name, arg_val]
            return args

        srun_argv: list[str] = sum(
            [
                _handle_arg(k)
                for k in type(self).fields().keys()
                if k not in (SlurmJobStep.fields().keys() | {"extra_argv"}) and getattr(self, k) is not None
            ],
            [],
        )

        return [str(arg) for arg in ["srun"] + srun_argv + self.extra_argv + self.command]


@dataclass
class EnrootJobStep(SrunJobStep):
    """Srun with enroot containers.

    See https://github.com/NVIDIA/enroot.
    """

    container_image: str | None = field(default=None)

    container_mounts: list[str] = field(default_factory=list)

    container_workdir: str | None = field(default=None)

    no_container_mount_home: bool = field(default=True)

    def __post_init__(self):
        if isinstance(self.container_mounts, list):
            self.container_mounts = ",".join(self.container_mounts)

        if not self.container_mounts:
            self.container_mounts = None

        super().__post_init__()


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

        return template.render(
            job_name=self.job_name,
            account=self.account,
            partition=self.partition,
            qos=self.qos,
            time=self.time,
            nodes=self.nodes,
            ntasks_per_node=self.ntasks_per_node,
            cpus_per_task=self.cpus_per_task,
            gpus_per_node=self.gpus_per_node,
            mem=self.mem,
            output=self.output,
            error=self.error,
            extra_argv=self.extra_argv,
            steps=self.steps,
        )
