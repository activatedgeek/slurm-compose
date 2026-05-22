import os
import tarfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from fsspec.generic import rsync

from .slurm import SlurmJob


@dataclass
class SlurmExporter:
    job: SlurmJob

    name: str | None = field(default=None)

    external_package_dirs: list[str | Path] = field(default_factory=list)

    export_dir: str | Path | None = field(default=None)

    home_dir: str | Path = field(
        default=Path(
            os.getenv(
                "SLURM_COMPOSE_HOME", Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "slurm-compose"
            )
        )
    )

    def __post_init__(self):
        self.name = self.name or self.job.job_name
        self.external_package_dirs = [Path(p).absolute() for p in self.external_package_dirs] + [
            Path(__file__).parents[1].absolute()
        ]

        self.export_dir = (
            Path(self.export_dir) if self.export_dir else Path(self.home_dir)
        ) / f"{self.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if self.export_dir.is_dir():
            raise RuntimeError(f"Export directory {self.export_dir} exists.")

        self.sbatch_file = self.export_dir / "sbatch.sh"
        self.package_dir = self.export_dir / "pkgs"

    def run(self, dry: bool = False, materialize_fn: Callable | None = None):
        if not dry:
            self.export_dir.mkdir(parents=True, exist_ok=False)

        if not dry:
            self.sbatch_file.write_text((materialize_fn or self.job.materialize)())
            self.sbatch_file.chmod(0o755)
        else:
            warnings.warn(f"Dry run. Skipping sbatch file {self.sbatch_file}.", RuntimeWarning)

        for package_dir in self.external_package_dirs:
            package_export_dir = self.package_dir / package_dir.name
            if not dry:
                rsync(str(package_dir), str(package_export_dir))
            else:
                warnings.warn(f"Dry run. Skipping sync {package_export_dir}.", RuntimeWarning)

        tar_file = self.export_dir / f"{self.export_dir.name}.tar.gz"
        if not dry:
            with tarfile.open(tar_file, "w:gz") as tar:
                tar.add(self.export_dir, arcname=self.export_dir.name)
        else:
            warnings.warn(f"Dry run. Skipping tar archive {tar_file}.", RuntimeWarning)
