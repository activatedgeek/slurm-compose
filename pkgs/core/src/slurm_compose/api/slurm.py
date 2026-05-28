from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from pkgutil import resolve_name
from typing import ClassVar, Iterator, Literal, Self, get_args, get_origin, get_type_hints

from jinja2 import Environment, FileSystemLoader
from ruamel.yaml import YAML

from slurm_compose.config import SBATCH_ERROR, SBATCH_OUTPUT

from .scripts import Script
from .scripts.utils import fields_to_argv, resolve_log_template


@dataclass
class SlurmJob:
    """Slurm Job Arguments

    All these arguments are passed to sbatch. See https://slurm.schedmd.com/sbatch.html for docs.

    `extra_argv` is a catch all for arguments that are currently part of the typed dataclass.
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

    array: str | None = field(default=None)

    extra_argv: list[str] = field(default_factory=list, metadata={"argv": False})
    """Must be equals separated to maintain template structure."""

    steps: list[Script] = field(default_factory=list, metadata={"argv": False})

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

    @classmethod
    def fields(cls) -> dict[str]:
        def _get_type(v):
            if get_origin(v) is ClassVar:
                return get_args(v)[0]
            return v

        return {k: _get_type(v) for k, v in get_type_hints(cls).items()}

    @classmethod
    def from_yaml(cls, file: str | Path) -> Iterator[Self]:
        with open(Path(file)) as f:
            yaml = YAML().load(f)

        for job_name, job_args in yaml.pop("jobs", {}).items():
            job_args["job_name"] = job_name

            steps = []
            for step_idx, step in enumerate(job_args.pop("steps", [])):
                if "__class__" not in step:
                    raise ValueError(f"Missing __class__ in step {step_idx}")

                step_cls = resolve_name(step.pop("__class__"))
                step = step_cls(**step)

                steps.append(step)

            yield cls(**job_args, steps=steps)

    def materialize(self, template: str = "slurm.sh.j2", template_dir: str | list[str | Path] | None = None) -> str:
        self.output = resolve_log_template(self.output, SBATCH_OUTPUT)
        self.error = resolve_log_template(self.error, SBATCH_ERROR)

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
            sbatch_argv=fields_to_argv(self, equals_separated=True) + (self.extra_argv or []),
            steps=self.steps,
        )
