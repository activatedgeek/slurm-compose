import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro
from rich.align import Align
from rich.panel import Panel
from rich.syntax import Syntax

from slurm_compose.api.exporter import SlurmExporter, SlurmSSHRemote
from slurm_compose.config import logger


@dataclass
class CLIConfig:
    file: Annotated[str | Path, tyro.conf.arg(aliases=["-f"])]
    """Path to slurm compose file."""

    host: Annotated[str | None, tyro.conf.arg(aliases=["-H"])] = field(default=None)
    """Name of remote host. Must be resolvable from ssh agent."""

    account: Annotated[str | None, tyro.conf.arg(aliases=["-A"])] = field(default=None)
    """sbatch account -A/--account."""

    partition: Annotated[str | None, tyro.conf.arg(aliases=["-p"])] = field(default=None)
    """sbatch partition -p/--partition."""

    qos: Annotated[str | None, tyro.conf.arg(aliases=["-q"])] = field(default=None)
    """sbatch qos -q/--qos."""

    time: Annotated[str | None, tyro.conf.arg(aliases=["-t"])] = field(default=None)
    """sbatch time -t/--time."""

    export_dir: Annotated[str | Path | None, tyro.conf.arg(aliases=["-d"])] = field(default=None)
    """Export directory."""

    dry: bool = field(default=False)
    """A dry run with no on-disk modifications when host is unset."""

    def __post_init__(self):
        self.exports: list[SlurmExporter] = []

        for export in SlurmExporter.from_yaml(self.file, export_dir=self.export_dir):
            export.job.account = self.account or export.job.account
            export.job.partition = self.partition or export.job.partition
            export.job.qos = self.qos or export.job.qos
            export.job.time = self.time or export.job.time

            self.exports.append(export)

    def run(self):
        if self.host:
            self.host = SlurmSSHRemote(self.host)

        for export in self.exports:
            info = export.sync(host=self.host, dry=self.dry)
            if self.dry:
                from slurm_compose.config import console

                if logger.getEffectiveLevel() == logging.DEBUG:
                    console.log(
                        Align.center(
                            Panel(
                                Syntax(info["sbatch"], "bash", line_numbers=True),
                                title=f"({export.job.job_name}) sbatch.sh",
                            )
                        )
                    )


def main():
    try:
        config = tyro.cli(CLIConfig, prog="slurm-compose")
        config.run()
    except Exception as e:
        logger.exception(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
