"""Validate and execute a generated Taskfile."""

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import tyro
from ruamel.yaml import YAML

from .logging import console
from .task import Taskfile


@dataclass
class TaskCommand:
    """Load a Taskfile specification and execute it with Task."""

    taskfile: Annotated[Path, tyro.conf.arg(aliases=["-t"])]
    """Path to the Taskfile YAML document."""

    output_dir: Annotated[Path | None, tyro.conf.arg(aliases=["-o"])] = None
    """Directory for generated files. Defaults to an automatically cleaned temporary directory."""

    force: Annotated[bool, tyro.conf.arg(aliases=["-f"])] = False
    """Overwrite an existing generated Taskfile.yml in the output directory."""

    silent: Annotated[bool, tyro.conf.arg(aliases=["-s"])] = True
    """Hide command source while streaming command output."""

    raw: Annotated[bool, tyro.conf.arg(aliases=["-r"])] = False
    """Print subprocess output literally without Rich logging decorations."""

    print_config: Annotated[bool, tyro.conf.arg(name="print")] = False
    """Print generated Taskfile YAML without executing it."""

    config: Taskfile = field(init=False)

    def __post_init__(self):
        yaml = YAML(typ="safe")
        yaml.allow_duplicate_keys = False
        self.config = Taskfile.model_validate(yaml.load(self.taskfile.read_text()))

    def __call__(self) -> int:
        if self.print_config:
            print(self.config.to_yaml(self.output_dir or Path.cwd()), end="")
            return 0

        with (
            TemporaryDirectory(prefix="stask-") if self.output_dir is None else nullcontext(self.output_dir)
        ) as output_dir:
            if self.output_dir is None:
                console.print(
                    f"Warning: Using temporary output directory {Path(output_dir).resolve()}; "
                    "it will be removed after execution.",
                    style="yellow",
                    markup=False,
                    soft_wrap=True,
                )

            with self.config.proc(
                output_dir,
                directory=self.taskfile.resolve().parent,
                silent=self.silent,
                raw=self.raw,
                force=self.force,
            ) as process:
                returncode = process.wait()

        return returncode


def main():
    try:
        exit_code = tyro.cli(TaskCommand, prog="stask")()
    except Exception as exc:
        console.print(f"Error: {exc}", style="red", markup=False)
        raise SystemExit(1) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
