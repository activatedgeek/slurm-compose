from dataclasses import dataclass, field

from .srun import SrunScript


@dataclass
class PyxisScript(SrunScript):
    """Srun with pyxis Slurm plugin arguments.

    See https://github.com/nvidia/pyxis.
    """

    container_image: str | None = field(default=None)

    container_mounts: str | list[str] = field(default_factory=list)

    container_workdir: str | None = field(default=None)

    container_mount_home: bool = field(default=False)

    def __post_init__(self):
        if not self.container_image:
            raise ValueError("container_image cannot be empty.")

        super().__post_init__()

    def pre_argv(self):
        super().pre_argv()

        if isinstance(self.container_mounts, str):
            self.container_mounts = self.container_mounts.split(",")
