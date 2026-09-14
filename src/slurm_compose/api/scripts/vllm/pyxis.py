from dataclasses import dataclass, field
from pathlib import Path

from slurm_compose.api.scripts import PyxisScript
from slurm_compose.config import MOUNT_PATH


@dataclass
class vLLMScript(PyxisScript):
    job_name: str = field(default="vllm")

    def __post_init__(self):
        super().__post_init__()

        self.command = [Path(MOUNT_PATH) / "pkgs/slurm_compose/api/scripts/vllm/srun.sh"] + self.command

        self.local_env["VLLM_RUNTIME_DIR"] = rf"$(mktemp -u {MOUNT_PATH}/logs/vllm-XXXXXXX)"
