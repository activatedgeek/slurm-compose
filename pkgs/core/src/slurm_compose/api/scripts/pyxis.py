from dataclasses import dataclass, field

from .srun import SrunScript


@dataclass
class PyxisScript(SrunScript):
    """Srun with pyxis Slurm plugin arguments.

    See https://github.com/nvidia/pyxis.
    """

    container_image: str | None = field(default=None)

    container_mounts: str | list[str] | None = field(default=None)

    container_workdir: str | None = field(default=None)

    container_mount_home: bool = field(default=False)

    def __post_init__(self):
        if not self.container_image:
            raise ValueError("container_image cannot be empty.")

        if isinstance(self.container_mounts, list):
            self.container_mounts = ",".join(self.container_mounts)

        if not self.container_mounts:
            self.container_mounts = None

        super().__post_init__()
