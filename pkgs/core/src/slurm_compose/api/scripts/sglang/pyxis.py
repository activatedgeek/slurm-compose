from dataclasses import dataclass, field
from pathlib import Path

from slurm_compose.api.scripts import PyxisScript
from slurm_compose.config import MOUNT_PATH


@dataclass
class SGLangScript(PyxisScript):
    job_name: str = field(default="sglang")

    def __post_init__(self):
        super().__post_init__()

        self.command = [Path(MOUNT_PATH) / "pkgs/slurm_compose/api/scripts/sglang/srun.sh"] + self.command
