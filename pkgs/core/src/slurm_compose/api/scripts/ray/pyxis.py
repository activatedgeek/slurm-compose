from dataclasses import dataclass, field
from pathlib import Path

from slurm_compose.api.scripts import PyxisScript
from slurm_compose.config import MOUNT_PATH


@dataclass
class RayScript(PyxisScript):
    job_name: str = field(default="ray")

    def __post_init__(self):
        super().__post_init__()

        self.command = [str(Path(MOUNT_PATH) / "pkgs/slurm_compose/api/scripts/ray/srun.sh")] + self.command

        self.env["RAY_RUNTIME_DIR"] = rf"$(mktemp -u {MOUNT_PATH}/logs/ray-XXXXXXX)"


@dataclass
class IdleRayScript(RayScript):
    """Only for coordination and not actual workloads."""

    nodes: int = field(default=1)

    ntasks_per_node: int = field(default=1)

    cpus_per_task: int = field(default=1)

    gpus_per_node: int = field(default=0)

    overlap: bool = field(default=True)

    def __post_init__(self):
        self.command = [
            "--num-cpus",
            1,
            "--num-gpus",
            0,
            "--object-store-memory",
            104857600,
            "--memory",
            104857600,
        ]

        self.overlap = True

        super().__post_init__()
