from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

from slurm_compose.api.exporter import SlurmExporter, SlurmSSHRemote
from slurm_compose.config import logger


@dataclass
class CLIConfig:
    file: Annotated[str | Path, tyro.conf.arg(aliases=["-f"])]
    """Path to template file."""

    job_names: Annotated[list[str], tyro.conf.Positional] = field(default_factory=list)
    """Optional space-separated list of job names to generate from template file."""

    data_files: Annotated[list[str | Path], tyro.conf.UseAppendAction, tyro.conf.arg(aliases=["-d"])] = field(
        default_factory=list
    )
    """Path to template data file for variable substitution. Repeatable. Last one wins."""

    data_args: Annotated[list[int | str], tyro.conf.UseAppendAction, tyro.conf.arg(aliases=["-e"])] = field(
        default_factory=list
    )
    """Inline template variables as VAR=value. Repeatable. Last one wins. Overrides -d/--data-files."""

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

    cpu: bool = field(default=False)
    """Use pre-registered cpu partitions from host configuration."""

    interactive: bool = field(default=False)
    """Use pre-registered interactive partitions from host configuration."""

    nodes: Annotated[int | None, tyro.conf.arg(aliases=["-N"])] = field(default=None)
    """sbatch nodes -N/--nodes."""

    gpus_per_node: Annotated[int | None, tyro.conf.arg(aliases=["-g"])] = field(default=None)
    """sbatch gpus per nodes --gpus-per-node."""

    array: Annotated[str | None, tyro.conf.arg(aliases=["-a"])] = field(default=None)
    """sbatch array -a/--array."""

    export_dir: str | Path | None = field(default=None)
    """Export directory."""

    dry: bool = field(default=False)
    """A dry run with no on-disk modifications when host is unset."""

    def run(self):
        self.exports: list[SlurmExporter] = list(
            SlurmExporter.from_yaml(
                self.file,
                job_names=self.job_names,
                data_files=self.data_files,
                data_kwargs=dict(arg.split("=", 1) for arg in self.data_args),
                export_dir=self.export_dir,
                job_kwargs={
                    "account": self.account,
                    "partition": self.partition,
                    "qos": self.qos,
                    "time": self.time,
                    "nodes": self.nodes,
                    "gpus_per_node": self.gpus_per_node,
                    "array": self.array,
                },
            )
        )

        if self.host:
            self.host = SlurmSSHRemote(self.host, interactive=self.interactive, cpu=self.cpu)
        else:
            logger.warning("Host not provided. sbatch-related configuration ignored.")

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
