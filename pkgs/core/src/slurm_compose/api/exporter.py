import os
import shlex
import shutil
import subprocess
import tarfile
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fsspec.implementations.sftp import SFTPFileSystem

from slurm_compose.config import CONFIG_HOME, EXPORTS_HOME, MOUNT_PATH, logger

from .packager import gitignore_filter
from .scripts import PyxisScript, SrunScript
from .slurm import SlurmJob


@dataclass
class SlurmSSHRemote:
    host: str

    config_file: str | Path | None = field(default=None)

    def __post_init__(self):
        self.config = SlurmSSHRemote.load_config(self.host, config_file=self.config_file)

        self.fs = SFTPFileSystem(
            host=self.config.get("hostname"),
            port=self.config.get("port"),
            username=self.config.get("user"),
            key_filename=self.config.get("identityfile"),
        )

        self.export_dir = Path(self.config["home_dir"]) / "exports"

    @staticmethod
    def load_config(host, config_file: str | Path = None) -> dict:
        host_config_file = Path(config_file or (CONFIG_HOME / "hosts.toml"))
        host_config = {}

        if host_config_file.exists():
            with open(host_config_file, "rb") as f:
                config = tomllib.load(f)

            if host in config:
                host_config = config[host]
            else:
                logger.warning(f"Missing config entry for {host}.")

        if host_config.pop("use_ssh_agent", True):
            logger.info(f"Loading {host} config from SSH agent.")
            host_config = {**SlurmSSHRemote.get_ssh_config(host), **host_config}

        assert "user" in host_config, f"Missing user in {host} config."

        if "home_dir" not in host_config:
            host_config["home_dir"] = f"/home/{host_config['user']}/.slurm-compose"
            logger.warning(f"Setting default host home directory {host_config['home_dir']}")

        return host_config

    @staticmethod
    def get_ssh_config(host):
        raw_ssh_config = dict(
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

        ssh_config = {
            "hostname": raw_ssh_config.get("hostname"),
            "port": int(raw_ssh_config.get("port", 22)),
            "user": raw_ssh_config.get("user"),
            "identityfile": raw_ssh_config.get("identityfile"),
        }

        if "identityfile" in ssh_config:
            ssh_config["identityfile"] = os.path.expanduser(ssh_config["identityfile"])
            if not Path(ssh_config["identityfile"]).exists():
                logger.warning(f"SSH identity file {ssh_config['identityfile']} not found. Ignoring.")
                ssh_config.pop("identityfile")

        return ssh_config

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

        if self.export_dir:
            self.export_dir = Path(self.export_dir)
        else:
            self.export_dir = EXPORTS_HOME
            logger.warning(f"Setting default exports home to {EXPORTS_HOME}.")

        self.export_dir = self.export_dir / f"{self.job.job_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if self.export_dir.exists():
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
            logger.info(f"Final bundle from {self.export_dir} synced to {host}:{sync_dir}")
            return sync_dir

        logger.info(f"Final bundle synced to {self.export_dir}")
        return self.export_dir
