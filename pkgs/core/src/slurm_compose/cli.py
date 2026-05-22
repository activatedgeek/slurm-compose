from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

from slurm_compose.api.exporter import SlurmExporter
from slurm_compose.api.slurm import SlurmJob


@dataclass
class CLIConfig:
    file: Annotated[str | Path, tyro.conf.arg(aliases=["-f"])]
    """Path to slurm compose file."""

    output: Annotated[str | Path | None, tyro.conf.arg(aliases=["-o"])] = field(default=None)
    """Path to slurm job stdout directory. Use to construct -o/--output."""

    error: Annotated[str | Path | None, tyro.conf.arg(aliases=["-e"])] = field(default=None)
    """Path to slurm job stderr directory. Use to construct -e/--error."""

    name: Annotated[str | None, tyro.conf.arg(aliases=["-n"])] = field(default=None)
    """Name of export."""

    export_dir: Annotated[str | Path | None, tyro.conf.arg(aliases=["-d"])] = field(default=None)
    """Export directory. Respects XDG_CACHE_HOME."""

    dry: bool = field(default=False)
    """A dry run with no on-disk modifications."""

    def __post_init__(self):
        self.jobs = SlurmJob.from_yaml(self.file, output=self.output, error=self.error)

    def run(self):
        self.exports = [
            SlurmExporter(
                job=job,
                name=self.name,
                export_dir=self.export_dir,
            )
            for job in self.jobs
        ]
        for export in self.exports:
            export.run(dry=self.dry)


def main():
    config = tyro.cli(CLIConfig, prog="slurm-compose")

    config.run()


if __name__ == "__main__":
    main()
