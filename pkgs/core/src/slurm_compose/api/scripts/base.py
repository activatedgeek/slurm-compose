from dataclasses import dataclass, field
from typing import ClassVar, get_args, get_origin, get_type_hints


@dataclass
class Script:
    """Slurm job step script.

    Sets up the command and environment to run in sbatch file.

    Any field marked with metadata argv=False is not used for automatic argv construction
    and must be handled separately.
    """

    command: str | list[str] = field(default_factory=list, metadata={"argv": False})

    env: dict[str, str] = field(default_factory=dict, metadata={"argv": False})

    def __post_init__(self):
        if not self.command:
            raise ValueError("command cannot be empty.")

    @classmethod
    def fields(cls) -> dict[str]:
        def _get_type(v):
            if get_origin(v) is ClassVar:
                return get_args(v)[0]
            return v

        return {k: _get_type(v) for k, v in get_type_hints(cls).items()}

    @property
    def argv(self) -> list[str]:
        if isinstance(self.command, str):
            raise ValueError("argv is not supported when command is a string.")

        return [str(arg) for arg in self.command]

    @property
    def args(self) -> str:
        if isinstance(self.command, str):
            return self.command

        return " ".join(self.argv)
