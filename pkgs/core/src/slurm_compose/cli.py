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
    """Path to slurm compose template file."""

    data_file: Annotated[str | Path | None, tyro.conf.arg(aliases=["-d"])] = field(default=None)
    """Path to slurm compose template data file for variable substitution."""

    data_args: Annotated[list[int | str], tyro.conf.arg(aliases=["-e"])] = field(default_factory=list)
    """Inline template variables as VAR=value. Repeatable; overrides values in data_file."""

    host: Annotated[str | None, tyro.conf.arg(aliases=["-H"])] = field(default=None)
    """Name of remote host. Must be resolvable from ssh agent."""

    account: Annotated[str | None, tyro.conf.arg(aliases=["-A"])] = field(default=None)
    """sbatch account -A/--account."""

    partition: Annotated[str | None, tyro.conf.arg(aliases=["-p"])] = field(default=None)
    """sbatch partition -p/--partition."""

    qos: Annotated[str | None, tyro.conf.arg(aliases=["-q"])] = field(default=None)
    """sbatch qos -q/--qos."""

    time: Annotated[str | None, tyro.conf.arg(aliases=["-t"])] = field(default=None)
    """sbatch time -t/--time. supports strings via pytimeparse."""

    interactive: bool = field(default=False)
    """Use pre-registered interactive partitions from host configuration."""

    cpu: bool = field(default=False)
    """Use pre-registered cpu partitions from host configuration."""

    export_dir: str | Path | None = field(default=None)
    """Export directory."""

    dry: bool = field(default=False)
    """A dry run with no on-disk modifications when host is unset."""

    def run(self):
        self.exports: list[SlurmExporter] = list(
            SlurmExporter.from_yaml(
                self.file,
                data_file=self.data_file,
                data_kwargs=dict(arg.split("=", 1) for arg in self.data_args),
                export_dir=self.export_dir,
                job_kwargs={
                    "account": self.account,
                    "partition": self.partition,
                    "qos": self.qos,
                    "time": self.time,
                },
            )
        )

        if self.host:
            self.host = SlurmSSHRemote(self.host, interactive=self.interactive, cpu=self.cpu)
        else:
            logger.warning("Host not provided. sbatch-related configuration ignored.")

        for export in self.exports:
            info = export.sync(host=self.host, dry=self.dry)
            if self.dry:
                from slurm_compose.config import console

                if logger.getEffectiveLevel() == logging.DEBUG:
                    console.log(
                        Align.center(
                            Panel(
                                Syntax(info["sbatch"], "bash", line_numbers=True, word_wrap=True),
                                title=f"({export.job.job_name}) sbatch.sh",
                            )
                        )
                    )
                else:
                    logger.info("Set SCOMPOSE_LOGLEVEL=DEBUG to view the materialized sbatch file.")


def main():
    try:
        config = tyro.cli(CLIConfig, prog="slurm-compose")
        config.run()
    except Exception as e:
        logger.exception(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
