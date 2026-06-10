from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from pkgutil import resolve_name
from typing import ClassVar, Literal, Self, get_args, get_origin, get_type_hints

from jinja2 import Environment, FileSystemLoader
from pytimeparse import parse as timeparse

from slurm_compose.config import SBATCH_ERROR, SBATCH_OUTPUT, logger
from slurm_compose.plugins.registry import script_plugins

from .scripts import Script
from .scripts.utils import fields_to_argv, maybe_update_fields, resolve_log_template


@dataclass
class SlurmJob:
    """Slurm Job Arguments

    All these arguments are passed to sbatch. See https://slurm.schedmd.com/sbatch.html for docs.

    `extra_argv` is a catch all for arguments that are currently part of the typed dataclass.

    Always use `maybe_update` method to ensure the fields are correctly in place, which calls
    the `pre_materialize` method (`materialize` always calls `pre_materialize`). `pre_materialize`
    method is idempotent.
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

    requeue: bool | str | None = field(default=True)

    array: str | None = field(default=None)

    extra_argv: list[str] = field(default_factory=list, metadata={"argv": False})
    """Must be equals separated to maintain template structure."""

    steps: list[Script] = field(default_factory=list, metadata={"argv": False})

    step_delay: int | str = field(default="30s", metadata={"argv": False})

    env: dict[str, str] = field(default_factory=dict, metadata={"argv": False})

    max_restarts: int = field(default=0, metadata={"argv": False})

    def __post_init__(self):
        if not self.job_name:
            raise ValueError("job_name cannot be empty.")

        self.pre_materialize()

    @classmethod
    def fields(cls) -> dict[str]:
        def _get_type(v):
            if get_origin(v) is ClassVar:
                return get_args(v)[0]
            return v

        return {k: _get_type(v) for k, v in get_type_hints(cls).items()}

    def to_dict(self) -> dict:
        def _parse_val(v):
            if v is None or isinstance(v, (int, str, float, bool)):
                return v
            elif isinstance(v, Path):
                return str(v)
            elif isinstance(v, list):
                return list(filter(lambda x: x is not None, [_parse_val(vi) for vi in v])) or None
            elif isinstance(v, dict):
                v_out = {k: _parse_val(vi) for k, vi in v.items()}
                return {k: vi for k, vi in v_out.items() if vi is not None} or None
            elif isinstance(v, Script):
                v_out = {k: _parse_val(getattr(v, k)) for k in v.fields().keys()} | {
                    "__class__": type(v).__module__ + ":" + type(v).__qualname__
                }
                return {k: vi for k, vi in v_out.items() if vi is not None} or None

            raise ValueError(f"Unsupported type {type(v).__module__ + ':' + type(v).__qualname__}")

        dict_config = {k: _parse_val(getattr(self, k)) for k in self.fields().keys()}

        return {k: v for k, v in dict_config.items() if v is not None}

    @classmethod
    def from_dict(cls, **kwargs: dict) -> Self:
        steps = []
        for step_idx, step in enumerate(kwargs.pop("steps", [])):
            if "__class__" in step and "step_type" in step:
                raise ValueError(f"Cannot specify both __class__ and step_type fields in step {step_idx}")

            if "__class__" not in step and "step_type" not in step:
                raise ValueError(f"Must specify one of __class__ or step_type fields in step {step_idx}")

            step_cls = (
                resolve_name(step.pop("__class__"))
                if "__class__" in step
                else script_plugins.get(step.pop("step_type"))()
            )
            try:
                step = step_cls(**step)
            except Exception:
                logger.error(f"Failed to build step {step_idx}")
                raise

            steps.append(step)

        return cls(**kwargs, steps=steps)

    def maybe_update(self, force: bool = False, **kwargs):
        maybe_update_fields(self, force=force, **kwargs)
        self.pre_materialize()

    def pre_materialize(self):
        self.output = resolve_log_template(self.output, SBATCH_OUTPUT)
        self.error = resolve_log_template(self.error, SBATCH_ERROR)

        ## Check for human-readable times.
        if isinstance(self.time, str):
            parsed_seconds = timeparse(self.time, granularity="seconds")
            if parsed_seconds:
                self.time = timedelta(seconds=parsed_seconds)

        if isinstance(self.time, timedelta):
            total_seconds = int(self.time.total_seconds())
            days, remainder = divmod(total_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            if days > 0:
                self.time = f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                self.time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        if isinstance(self.gpus_per_node, int) and self.gpus_per_node < 0:
            self.gpus_per_node = None

        if not self.requeue:
            self.max_restarts = 0

    def materialize(self, template: str = "slurm.sh.j2", template_dir: str | list[str | Path] | None = None) -> str:
        self.pre_materialize()

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
            env=self.env,
            steps=self.steps,
            step_delay=self.step_delay,
            max_restarts=self.max_restarts,
        )
