from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro


@dataclass
class CLIConfig:
    file: Annotated[str | Path, tyro.conf.arg(aliases=["-f"])]
    """Path to slurm compose file."""

    template: Annotated[str | Path, tyro.conf.arg(aliases=["-t"])] = field(
        default=Path(__file__).parent / "templates/slurm.sh.j2"
    )
    """Path to slurm template."""

    def __post_init__(self):
        self.file = Path(self.file)

        self.template = Path(self.template)


def main():
    config = tyro.cli(CLIConfig, prog="slurm-compose")

    print(config)


if __name__ == "__main__":
    main()
