import logging
import os
from pathlib import Path

from rich.logging import RichHandler

HOME = Path(os.getenv("SCOMPOSE_HOME") or Path.cwd() / ".slurm-compose")
EXPORTS_HOME = HOME / "exports"

CONFIG_HOME = Path(
    os.getenv("SCOMPOSE_CONFIG_HOME") or Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "slurm-compose"
)

PROJECT_NAME = os.getenv("SCOMPOSE_PROJECT_NAME")

MOUNT_PATH = os.getenv("SCOMPOSE_MOUNT_PATH", "/sc")

SBATCH_OUTPUT = os.getenv("SCOMPOSE_SBATCH_OUTPUT", r"%j-%x.log")
SBATCH_ERROR = os.getenv("SCOMPOSE_SBATCH_ERROR", r"%j-%x.err")
SRUN_OUTPUT = os.getenv("SCOMPOSE_SRUN_OUTPUT", r"%j.%s-${STEP_NAME}.log")
SRUN_ERROR = os.getenv("SCOMPOSE_SRUN_ERROR", r"%j.%s-${STEP_NAME}.err")

logger = logging.getLogger("slurm-compose")
logger.propagate = False
logger.addHandler(RichHandler(markup=True))
logger.setLevel(os.getenv("SCOMPOSE_LOGLEVEL", "INFO"))
try:
    console = logger.handlers[0].console
except AttributeError:
    ...
