from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

from slurm_compose.api.exporter import SlurmExporter
from slurm_compose.api.slurm import SlurmJob
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
        self.jobs = []

        for job in SlurmJob.from_yaml(self.file):
            job.account = self.account or job.account
            job.partition = self.partition or job.partition
            job.qos = self.qos or job.qos
            job.time = self.time or job.time

            self.jobs.append(job)

    def run(self):
        self.exports = [SlurmExporter(job=job, export_dir=self.export_dir) for job in self.jobs]
        for export in self.exports:
            export.sync(host=self.host, dry=self.dry)


def main():
    try:
        config = tyro.cli(CLIConfig, prog="slurm-compose")
        config.run()
    except Exception as e:
        logger.exception(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
