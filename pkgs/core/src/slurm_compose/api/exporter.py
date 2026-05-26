import os
import shlex
import subprocess
import tarfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Callable

from fsspec.generic import rsync
from fsspec.implementations.sftp import SFTPFileSystem

from .slurm import SlurmJob


@dataclass
class SlurmRemote:
    host: str

    home_dir: str | Path = field(default_factory=lambda: os.getenv("SLURM_COMPOSE_REMOTE_HOME"))

    def __post_init__(self):
        self.ssh_config = dict(
            [
                tuple(line.split(maxsplit=1))
                for line in subprocess.run(
                    ["ssh", "-G", self.host],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.split("\n")
                if line
            ]
        )

        self.fs = SFTPFileSystem(
            host=self.ssh_config.get("hostname"),
            port=int(self.ssh_config.get("port", 22)),
            username=self.ssh_config.get("user"),
            key_filename=os.path.expanduser(self.ssh_config.get("identityfile")),
        )

        self.home_dir = Path(self.home_dir) if self.home_dir else (self.HOME / ".slurm-compose")
        self.export_dir = self.home_dir / "exports"

    @cached_property
    def HOME(self) -> Path:
        _, stdout, _ = self.exec("echo $HOME")
        return Path(stdout.read().decode().strip())

    def exec(self, *args, **kwargs):
        stdin, stdout, stderr = self.fs.client.exec_command(*args, **kwargs)
        if stdout.channel.recv_exit_status():
            raise RuntimeError(f"Failed exec on {self.host}.")

        return stdin, stdout, stderr

    def sync(self, local_dir: str | Path) -> Path:
        local_dir = Path(local_dir)

        tar_file = local_dir.parent / f"{local_dir.name}.tar.gz"
        with tarfile.open(tar_file, "w:gz") as tar:
            tar.add(local_dir, arcname=local_dir.name)

        try:
            self.fs.mkdir(str(self.export_dir), create_parents=True, mode=0o700)
        except FileExistsError:
            pass

        self.fs.put(str(tar_file), str(self.export_dir / tar_file.name))

        self.exec(
            " && ".join(
                [
                    shlex.join(
                        [
                            "tar",
                            "-xzf",
                            str(self.export_dir / tar_file.name),
                            "-C",
                            str(self.export_dir),
                        ]
                    ),
                    shlex.join(["rm", str(self.export_dir / tar_file.name)]),
                ]
            )
        )

        return self.export_dir / local_dir.name


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

    def sync(
        self,
        host: str = None,
        host_dir: str | Path | None = None,
        dry: bool = False,
        materialize_fn: Callable | None = None,
    ):
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

        if not dry:
            if host:
                remote = SlurmRemote(host, home_dir=host_dir)
                return remote.sync(self.export_dir)
            else:
                return self.export_dir
        else:
            warnings.warn(f"Dry run. Skipping sync to {host}.", RuntimeWarning)
