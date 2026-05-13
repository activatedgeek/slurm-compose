from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

from slurm_compose.api.slurm import SlurmJob


@dataclass
class CLIConfig:
    file: Annotated[str | Path, tyro.conf.arg(aliases=["-f"])]
    """Path to slurm compose file."""

    output: Annotated[str | Path | None, tyro.conf.arg(aliases=["-o"])] = field(default=None)
    """Path to slurm job stdout directory. Use to construct -o/--output."""

    error: Annotated[str | Path | None, tyro.conf.arg(aliases=["-e"])] = field(default=None)
    """Path to slurm job stderr directory. Use to construct -e/--error."""

    write: Annotated[str | Path | None, tyro.conf.arg(aliases=["-w"])] = field(default=None)
    """Path to write materialized sbatch files."""

    def __post_init__(self):
        self.file = Path(self.file)

        self.jobs = SlurmJob.from_yaml(self.file, output=self.output, error=self.error)

    def run(self):
        if not self.write:
            return

        self.write = Path(self.write)
        self.write.mkdir(exist_ok=True, parents=True)

        for job in self.jobs:
            sbatch_file = self.write / f"{job.job_name}_sbatch.sh"
            sbatch_file.write_text(job.materialize())
            sbatch_file.chmod(0o755)


def main():
    config = tyro.cli(CLIConfig, prog="slurm-compose")

    config.run()


if __name__ == "__main__":
    main()
