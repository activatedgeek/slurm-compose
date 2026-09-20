import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import tyro
from ruamel.yaml import YAML

from slurm_compose import __version__
from slurm_compose.config import logger
from slurm_compose.spec import ComposeConfig, SbatchJobConfig


@dataclass
class ConfigCommand:
    """Parse, resolve, and render scompose file in canonical format"""

    file: Annotated[Path, tyro.conf.arg(aliases=["-f"])]
    """Compose configuration file."""

    job_name: Annotated[str | None, tyro.conf.arg(aliases=["-J"])] = None
    """Override the Compose project name."""

    jobs_dir: Annotated[Path | None, tyro.conf.arg(aliases=["-d"])] = None
    """Override the Compose state directory."""

    vars: Annotated[list[str], tyro.conf.arg(name="var"), tyro.conf.UseAppendAction] = field(default_factory=list)
    """Override variables using NAME=VALUE syntax."""

    compose: ComposeConfig = field(init=False)

    def __post_init__(self):
        runtime_vars = {}
        yaml = YAML(typ="safe")
        for assignment in self.vars:
            key, separator, value = assignment.partition("=")
            if not separator or not key:
                raise ValueError(f"runtime variable must use NAME=VALUE syntax: {assignment!r}")
            runtime_vars[key] = yaml.load(value)
        self.compose = ComposeConfig.from_yaml(self.file, self.job_name, self.jobs_dir, runtime_vars)

    def __call__(self):
        print(self.compose.yaml(), end="")


@dataclass
class UpCommand(ConfigCommand):
    """Create and start job."""

    passthrough: Annotated[list[str], tyro.conf.Positional] = field(default_factory=list)
    """Arguments after '--' separator are passed through unchanged."""

    sbatch: SbatchJobConfig = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.sbatch = SbatchJobConfig(compose=self.compose, passthrough=self.passthrough)

    def __call__(self):
        print(self.sbatch.script(), end="")


@dataclass
class VersionCommand:
    """Print the installed Slurm Compose version."""

    format: Annotated[Literal["pretty", "json"], tyro.conf.arg(aliases=["-f"])] = "pretty"
    """Version output format."""

    short: Annotated[bool, tyro.conf.arg(aliases=["-s"])] = False
    """Print only the version number."""

    def __call__(self):
        if self.short:
            print(__version__)
        elif self.format == "json":
            print(json.dumps({"version": __version__}))
        else:
            print(f"Slurm Compose version {__version__}")


CLICommand = (
    Annotated[ConfigCommand, tyro.conf.subcommand(name="config", prefix_name=False)]
    | Annotated[UpCommand, tyro.conf.subcommand(name="up", prefix_name=False)]
    | Annotated[VersionCommand, tyro.conf.subcommand(name="version", prefix_name=False)]
)


def main():
    try:
        tyro.cli(CLICommand, prog="scompose")()
    except Exception as exc:
        logger.exception(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
