import os
from pathlib import Path

HOME = Path(os.getenv("SCOMPOSE_HOME") or Path.cwd() / ".slurm-compose")
EXPORTS_HOME = HOME / "exports"

CONFIG_HOME = Path(os.getenv("SCOMPOSE_CONFIG_HOME") or HOME / "config")

MOUNT_PATH = os.getenv("SCOMPOSE_MOUNT_PATH", "/sc")

SBATCH_OUTPUT = os.getenv("SCOMPOSE_SBATCH_OUTPUT", "%j-%x.log")
SBATCH_ERROR = os.getenv("SCOMPOSE_SBATCH_ERROR", "%j-%x.err")
SRUN_OUTPUT = os.getenv("SCOMPOSE_SRUN_OUTPUT", "%j-%s.%x.log")
SRUN_ERROR = os.getenv("SCOMPOSE_SRUN_ERROR", "%j-%s.%x.err")
