from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import ClassVar, Literal, Self

from jinja2 import Environment, FileSystemLoader
from ruamel.yaml import YAML

from .base import BaseArgs
from .utils import fields_to_argv, resolve_log_template


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
class SlurmJob(BaseArgs):
    """Slurm Job Arguments

    All these arguments are passed to sbatch. See https://slurm.schedmd.com/sbatch.html for docs.

    `extra_argv` is a catch all for arguments that are currently part of the typed dataclass.
    """

    _output_template: ClassVar[str] = "%j-%x.log"

    _error_template: ClassVar[str] = "%j-%x.err"

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
        if not self.job_name:
            raise ValueError("job_name cannot be empty.")

        if isinstance(self.time, timedelta):
            total_seconds = int(self.time.total_seconds())
            days, remainder = divmod(total_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            if days > 0:
                self.time = f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                self.time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        self.output = resolve_log_template(self.output, self._output_template)
        self.error = resolve_log_template(self.error, self._error_template) or self.output

    def materialize(self, template: str = "slurm.sh.j2", template_dir: str | list[str | Path] | None = None) -> str:
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
            self,
            ignore_keys=BaseArgs.fields().keys() | {"extra_argv", "steps", "_output_template", "_error_template"},
            equals_separated=True,
        )

        return template.render(
            sbatch_argv=sbatch_argv + (self.extra_argv or []),
            steps=self.steps,
        )

    @classmethod
    def from_yaml(
        cls, file: str | Path, output: str | Path | None = None, error: str | Path | None = None
    ) -> list[Self]:
        with open(Path(file)) as f:
            yaml = YAML().load(f)

        jobs = []
        for job_name, job_args in yaml.pop("jobs", {}).items():
            job_args["job_name"] = job_name
            job_args["output"] = output
            job_args["error"] = error

            steps = []
            for step_idx, step in enumerate(job_args.pop("steps", [])):
                ## Auto-infer job step class. FIXME.
                from slurm_compose.api import PyxisJobStep, SlurmJobStep, SrunJobStep

                for step_cls in [PyxisJobStep, SrunJobStep, SlurmJobStep]:
                    step_cls_keys = step_cls.fields().keys()
                    if step.keys() & step_cls_keys:
                        step_cls_args = {**step}
                        if {"output", "error"} & step_cls_keys:
                            step_cls_args.update({"output": output, "error": error})

                        step = step_cls(**step_cls_args)
                        break

                if not isinstance(step, SlurmJobStep):
                    raise ValueError(f"Unable to parse step {step_idx}.")

                steps.append(step)

            job = cls(**job_args, steps=steps)
            jobs.append(job)

        return jobs
