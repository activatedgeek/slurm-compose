import os
import shlex
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

from fsspec.implementations.sftp import SFTPFileSystem

from slurm_compose.config import EXPORTS_HOME, MOUNT_PATH, logger

from .packager import gitignore_filter
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

        if self.home_dir:
            self.home_dir = Path(self.home_dir)
        else:
            self.home_dir = self.HOME / ".slurm-compose"
            logger.warning(f"Setting default host home directory {self.host}:{self.home_dir}")

        self.export_dir = self.home_dir / "exports"

    @staticmethod
    def get_ssh_config(host):
        ssh_config = dict(
            [
                tuple(line.split(maxsplit=1))
                for line in subprocess.run(
                    ["ssh", "-q", "-G", host],
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
                logger.warning(f"SSH identity file {ssh_config['identityfile']} not found. Ignoring.")
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

    def sync(self, local_dir: str | Path, dry: bool = False) -> Path:
        local_dir = Path(local_dir)
        sync_meta_dir = local_dir / ".sync"

        ## Pre-sync cleanup to create tar file.
        if sync_meta_dir.exists():
            logger.warning(f"Deleted existing sync directory {sync_meta_dir}.")
            shutil.rmtree(sync_meta_dir)

        ## Create tar to copy (better than copying single files).
        tar_file = local_dir.parent / f"{local_dir.name}.tar.gz"
        if dry:
            logger.warning(f"Dry run. Skipping tar file creation at {tar_file}.")
        else:
            with tarfile.open(tar_file, "w:gz") as tar:
                tar.add(local_dir, arcname=local_dir.name)

        if not dry:
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

        logger.info(f"Directory synced to {self.host}:{self.export_dir / local_dir.name}")

        return self.export_dir / local_dir.name


@dataclass
class SlurmExporter:
    job: SlurmJob

    name: str | None = field(default=None)

    external_package_dirs: list[str | Path] = field(default_factory=list)

    export_dir: str | Path | None = field(default=None)

    def __post_init__(self):
        self.external_package_dirs = [Path(p).absolute() for p in self.external_package_dirs] + [
            Path(__file__).parents[1].absolute()
        ]

        self.export_dir = (
            Path(self.export_dir) if self.export_dir else EXPORTS_HOME
        ) / f"{self.job.job_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if self.export_dir.is_dir():
            raise RuntimeError(f"Export directory {self.export_dir} exists.")

        self.sbatch_file = self.export_dir / "sbatch.sh"
        self.package_dir = self.export_dir / "pkgs"

    def bundle(self, mount_dir: str | Path | None = None, dry: bool = False):
        mount_dir = mount_dir or self.export_dir
        mount_spec = f"{mount_dir}:{MOUNT_PATH}"

        ## Set sbatch output.
        self.job.job_name = self.export_dir.name
        self.job.output = mount_dir / "logs"

        for step_idx, step in enumerate(self.job.steps):
            ## Set bundle mount when available.
            if isinstance(step, PyxisScript) and mount_spec not in step.container_mounts:
                step.container_mounts += [mount_spec]
                logger.debug(
                    f"Mount {step.container_mounts[-1]} added to {self.job.job_name} at step {step.job_name} (index {step_idx})"
                )

            ## Set output to logs path.
            if isinstance(step, SrunScript):
                step.output = Path(mount_dir) / "logs"
                logger.debug(
                    f"Output {step.output} set to {self.job.job_name} at step {step.job_name} (index {step_idx})"
                )

        ## Create local directory for export.
        if not dry:
            self.export_dir.mkdir(parents=True, exist_ok=False)

        ## Materialize sbatch file.
        if dry:
            logger.warning(f"Dry run. Skipping sbatch materialization at {self.sbatch_file}.")
        else:
            self.sbatch_file.write_text(self.job.materialize())
            self.sbatch_file.chmod(0o755)

        ## Materialize packages to bundle together.
        for package_dir in self.external_package_dirs:
            package_export_dir = self.package_dir / package_dir.name
            if dry:
                logger.warning(f"Dry run. Skipping package sync to {package_export_dir}.")
            else:
                shutil.copytree(
                    package_dir,
                    package_export_dir,
                    ignore=gitignore_filter(
                        package_dir,
                        ignore_files=[
                            Path(__file__).parent / "python.scignore",
                            package_dir / ".gitignore",
                            package_dir / ".scignore",
                        ],
                        ignore_patterns=[".git/"],
                    ),
                )

    def sync(self, host: str = None, dry: bool = False):
        if host:
            remote = SlurmSSHRemote(host)

        self.bundle(mount_dir=remote.export_dir / self.export_dir.name if host else None, dry=dry)

        if host:
            sync_dir = remote.sync(self.export_dir, dry=dry)
            logger.info(f"Final bundle synced to {host}:{sync_dir}")
            return sync_dir

        logger.info(f"Final bundle synced to {self.export_dir}")
        return self.export_dir
