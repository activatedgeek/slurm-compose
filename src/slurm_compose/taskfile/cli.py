"""Validate and execute a generated Taskfile."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import tyro
from ruamel.yaml import YAML

from .spec import Taskfile


@dataclass
class TaskCommand:
    """Load a Taskfile specification and execute it with Task."""

    taskfile: Annotated[Path, tyro.conf.arg(aliases=["-t"])]
    """Path to the Taskfile YAML document."""

    config: Taskfile = field(init=False)

    def __post_init__(self):
        yaml = YAML(typ="safe")
        yaml.allow_duplicate_keys = False
        self.config = Taskfile.model_validate(yaml.load(self.taskfile.read_text()))
        breakpoint()

    def __call__(self) -> int:
        with TemporaryDirectory(prefix="stask-") as directory:
            generated = Path(directory) / "Taskfile.yml"
            generated.write_text(self.config.to_yaml())
            result = subprocess.run(
                [
                    "task",
                    "--exit-code",
                    "--dir",
                    str(self.taskfile.resolve().parent),
                    "--taskfile",
                    str(generated),
                ],
                check=False,
            )
        return result.returncode


def main():
    raise SystemExit(tyro.cli(TaskCommand)())


if __name__ == "__main__":
    main()
