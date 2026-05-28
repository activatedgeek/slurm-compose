import os
import shlex
import shutil
import subprocess
import tarfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

from fsspec.implementations.sftp import SFTPFileSystem

from slurm_compose.config import EXPORTS_HOME, MOUNT_PATH

from .scripts import PyxisScript, SrunScript
from .slurm import SlurmJob


@dataclass
class SlurmSSHRemote:
    host: str

    ## FIXME: read all additional host config from CONFIG_HOME.
    home_dir: str | Path | None = field(default=os.getenv("SCOMPOSE_REMOTE_HOME"))

    def __post_init__(self):
        self.ssh_config = SlurmSSHRemote.get_ssh_config(self.host)

        self.fs = SFTPFileSystem(
            host=self.ssh_config.get("hostname"),
            port=self.ssh_config.get("port"),
            username=self.ssh_config.get("user"),
            key_filename=self.ssh_config.get("identityfile"),
        )

        self.home_dir = Path(self.home_dir) if self.home_dir else (self.HOME / ".slurm-compose")
        self.export_dir = self.home_dir / "exports"

    @staticmethod
    def get_ssh_config(host):
        ssh_config = dict(
            [
                tuple(line.split(maxsplit=1))
                for line in subprocess.run(
                    ["ssh", "-G", host],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.split("\n")
                if line
            ]
        )

        if "port" in ssh_config:
            ssh_config["port"] = int(ssh_config["port"])

        if "identityfile" in ssh_config:
            ssh_config["identityfile"] = os.path.expanduser(ssh_config["identityfile"])
            if not Path(ssh_config["identityfile"]).exists():
                ssh_config.pop("identityfile")

        return ssh_config

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
        sync_meta_dir = local_dir / ".sync"

        ## Pre-sync cleanup to create tar file.
        if sync_meta_dir.exists():
            shutil.rmtree(sync_meta_dir)

        ## Create tar to copy (better than copying single files).
        tar_file = local_dir.parent / f"{local_dir.name}.tar.gz"
        with tarfile.open(tar_file, "w:gz") as tar:
            tar.add(local_dir, arcname=local_dir.name)

        ## Ensure remote export directory.
        try:
            self.fs.mkdir(str(self.export_dir), create_parents=True, mode=0o700)
        except FileExistsError:
            ...

        ## Copy and untar to remote export directory.
        try:
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
        except:
            raise
        finally:
            tar_file.unlink()

        ## Post-sync metadata.
        sync_meta_dir.mkdir()
        with open(local_dir / ".sync" / self.host, "w") as f:
            f.write(str(self.export_dir / local_dir.name))

        return self.export_dir / local_dir.name


@dataclass
class SlurmExporter:
    job: SlurmJob

    name: str | None = field(default=None)

    external_package_dirs: list[str | Path] = field(default_factory=list)

    export_dir: str | Path | None = field(default=None)

    def __post_init__(self):
        self.name = self.name or self.job.job_name
        self.external_package_dirs = [Path(p).absolute() for p in self.external_package_dirs] + [
            Path(__file__).parents[1].absolute()
        ]

        self.export_dir = (
            Path(self.export_dir) if self.export_dir else EXPORTS_HOME
        ) / f"{self.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if self.export_dir.is_dir():
            raise RuntimeError(f"Export directory {self.export_dir} exists.")

        self.sbatch_file = self.export_dir / "sbatch.sh"
        self.package_dir = self.export_dir / "pkgs"

    def bundle(self, mount_dir: str | Path | None = None, dry: bool = False):
        mount_dir = mount_dir or self.export_dir
        mount_spec = f"{mount_dir}:{MOUNT_PATH}"

        ## Set sbatch output.
        self.job.output = mount_dir / "logs"

        for step in self.job.steps:
            ## Set bundle mount when available.
            if isinstance(step, PyxisScript) and mount_spec not in step.container_mounts:
                step.container_mounts += [mount_spec]

            ## Set output to logs path.
            if isinstance(step, SrunScript):
                step.output = Path(mount_dir) / "logs"

        ## Create local directory for export.
        if not dry:
            self.export_dir.mkdir(parents=True, exist_ok=False)

        ## Materialize sbatch file.
        if not dry:
            self.sbatch_file.write_text(self.job.materialize())
            self.sbatch_file.chmod(0o755)
        else:
            warnings.warn(f"Dry run. Skipping sbatch file {self.sbatch_file}.", RuntimeWarning)

        ## Materialize packages to bundle together.
        for package_dir in self.external_package_dirs:
            package_export_dir = self.package_dir / package_dir.name
            if not dry:
                shutil.copytree(package_dir, package_export_dir)
            else:
                warnings.warn(f"Dry run. Skipping sync {package_export_dir}.", RuntimeWarning)

    def sync(self, host: str = None, dry: bool = False):
        if host:
            remote = SlurmSSHRemote(host)

        self.bundle(mount_dir=remote.export_dir / self.export_dir.name if host else None, dry=False if host else dry)

        if host:
            if not dry:
                return remote.sync(self.export_dir)
            else:
                warnings.warn(f"Dry run. Skipping sync to {host}.", RuntimeWarning)

        return self.export_dir
