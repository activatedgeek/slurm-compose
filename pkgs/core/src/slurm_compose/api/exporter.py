import os
import shlex
import shutil
import subprocess
import tarfile
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Iterator, Self

from fsspec.implementations.sftp import SFTPFileSystem
from ruamel.yaml import YAML

from slurm_compose.config import CONFIG_HOME, EXPORTS_HOME, MOUNT_PATH, PROJECT_NAME, logger

from .packager import gitignore_filter
from .scripts import PyxisScript, SrunScript
from .slurm import SlurmJob


@dataclass
class SlurmSSHRemote:
    host: str

    config_file: str | Path | None = field(default=None)

    hostname: str | None = field(default=None)

    port: int | None = field(default=None)

    user: int | None = field(default=None)

    identityfile: str | Path | None = field(default=None)

    home_dir: str | Path | None = field(default=None)

    interactive: bool = field(default=False)

    cpu: bool = field(default=False)

    def __post_init__(self):
        config = SlurmSSHRemote.load_config(self.host, config_file=self.config_file)

        self.hostname = self.hostname or config.get("hostname")
        self.port = self.port or config.get("port")
        self.user = self.user or config.get("user")
        self.identityfile = self.identityfile or config.get("identityfile")
        self.home_dir = self.home_dir or config.get("home_dir")
        self.sbatch_config = config.get("sbatch")

        self.export_dir = Path(self.home_dir) / "exports"

        self.fs = SFTPFileSystem(
            host=self.hostname,
            port=self.port,
            username=self.user,
            key_filename=self.identityfile,
        )

    @property
    def partition(self) -> dict:
        config_name = ("cpu" if self.cpu else "gpu") + ("_interactive" if self.interactive else "")
        if config_name not in self.sbatch_config.get("partitions", {}):
            raise ValueError(f"Partition configuration {config_name} not found for host {self.host}.")

        return self.sbatch_config["partitions"][config_name]

    @property
    def gpus_per_node(self) -> int | None:
        return self.sbatch_config.get("gpus_per_node")

    @staticmethod
    def load_config(host, config_file: str | Path = None) -> dict:
        host_config_file = Path(config_file or (CONFIG_HOME / "hosts.toml"))
        host_config = {}

        if host_config_file.exists():
            with open(host_config_file, "rb") as f:
                config = tomllib.load(f)

            if host in config:
                host_config = config[host]

                if host_config.get("type", "ssh") == "alias":
                    if host_config["remote"] in config:
                        host_config = config[host_config["remote"]]
                    else:
                        logger.warning(f"Missing config entry for resolved alias host {host_config['remote']}.")
            else:
                logger.warning(f"Missing config entry for {host}.")

        if host_config.pop("use_ssh_agent", True):
            logger.info(f"Loading {host} config from SSH agent.")
            host_config = SlurmSSHRemote.get_ssh_config(host) | host_config

        assert "user" in host_config, f"Missing user in {host} config."

        if "home_dir" not in host_config:
            host_config["home_dir"] = f"/home/{host_config['user']}/.slurm-compose"
            logger.warning(f"Setting default host home directory {host_config['home_dir']}")

        host_config["sbatch"] = host_config.get("sbatch", {})
        host_config["sbatch"]["bin"] = host_config["sbatch"].get("bin", "sbatch")

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
            raise RuntimeError(f"Failed exec on {self.host}: {stderr.read().decode()}.")

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

            logger.info(f"Copying {tar_file} to {self.hostname} ({os.path.getsize(tar_file) / (1024**2):.2f} MB)")

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

    def submit(self, sbatch_file: str | Path) -> int:
        _, stdout, _ = self.exec(
            " ".join(
                [
                    str(self.sbatch_config["bin"]),
                    "--parsable",
                    str(sbatch_file),
                ]
            )
        )

        return int(stdout.read().decode())


@dataclass
class SlurmExporter:
    job: SlurmJob

    name: str | None = field(default=PROJECT_NAME)

    external_package_dirs: list[str | Path] = field(default_factory=list)

    export_dir: str | Path | None = field(default=None)

    def __post_init__(self):
        self.name = f"{self.name}-{self.job.job_name}" if self.name else self.job.job_name

        self.external_package_dirs = [Path(p).resolve() for p in self.external_package_dirs] + [
            Path(__file__).parents[1]
        ]

        if self.export_dir:
            self.export_dir = Path(self.export_dir)
        else:
            self.export_dir = EXPORTS_HOME
            logger.warning(f"Setting default exports home to {EXPORTS_HOME}.")

        self.export_dir = self.export_dir / f"{self.name}.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if self.export_dir.exists():
            raise RuntimeError(f"Export directory {self.export_dir} exists.")

        self.sbatch_file = self.export_dir / "sbatch.sh"
        self.package_dir = self.export_dir / "pkgs"

    @classmethod
    def from_yaml(
        cls,
        file: str | Path,
        data_file: str | Path | None = None,
        data_kwargs: dict | None = None,
        job_kwargs: dict | None = None,
        **kwargs,
    ) -> Iterator[Self]:
        job_kwargs = {k: v for k, v in (job_kwargs or {}).items() if v is not None}
        data_file_kwargs = {
            k: v for k, v in (YAML().load(Path(data_file).read_text()) if data_file else {}).items() if v is not None
        }
        data_kwargs = {k: v for k, v in (data_kwargs or {}).items() if v is not None}

        ## Apply template variable substitution before parsing YAML.
        yaml = YAML().load(Template(Path(file).read_text()).safe_substitute(**(data_file_kwargs | data_kwargs)))

        version = str(yaml.pop("version", 1))
        if version == "1":
            external_package_dirs = yaml.pop("external_package_dirs", []) + kwargs.pop("external_package_dirs", [])

            for job_name, job_args in yaml.pop("jobs", {}).items():
                ## Always respect job_name from YAML config.
                job_args.pop("job_name", None)
                job_kwargs.pop("job_name", None)

                yield cls(
                    job=SlurmJob.from_dict(job_name=job_name, **(job_args | job_kwargs)),
                    external_package_dirs=external_package_dirs,
                    **kwargs,
                )
        else:
            raise NotImplementedError(f"Unsupported yaml config version {version}")

    def bundle(self, host: SlurmSSHRemote | None = None, dry: bool = False) -> str:
        ####### WARNING: Bundling introduces side-effects to the self.job object. #######

        ## Set job sbatch params and apply overrides from host config when available.
        force_updates = {"job_name": self.export_dir.name}
        maybe_updates = {}

        mount_dir = self.export_dir

        if host:
            mount_dir = host.export_dir / self.export_dir.name

            partition = host.partition

            force_updates |= partition.pop("overrides", {})

            maybe_updates["account"] = host.sbatch_config.get("account")
            maybe_updates |= partition

            if not host.cpu:
                maybe_updates |= {"gpus_per_node": host.gpus_per_node}

        force_updates["output"] = mount_dir / "logs"

        self.job.maybe_update(**maybe_updates)
        self.job.maybe_update(**force_updates, force=True)

        self.job.env["SCOMPOSE_JOB"] = "1"
        self.job.env["SCOMPOSE_PKGS"] = f"{mount_dir}/pkgs"
        self.job.env["SCOMPOSE_LOGS"] = f"{mount_dir}/logs"

        ## Remove items no longer necessary for steps.
        [force_updates.pop(k, None) for k in ["job_name"]]
        [maybe_updates.pop(k, None) for k in ["account", "partition", "qos", "time"]]

        for step_idx, step in enumerate(self.job.steps):
            if isinstance(step, SrunScript):
                step.maybe_update(**maybe_updates)
                step.maybe_update(**force_updates, force=True)

            ## Set bundle mount when available.
            mount_spec = f"{mount_dir}:{MOUNT_PATH}"
            if isinstance(step, PyxisScript) and mount_spec not in step.container_mounts:
                self.job.env["SCOMPOSE_PKGS"] = f"{MOUNT_PATH}/pkgs"
                self.job.env["SCOMPOSE_LOGS"] = f"{MOUNT_PATH}/logs"
                step.container_mounts += [mount_spec]
                logger.debug(
                    f"Mount {step.container_mounts[-1]} added to {self.job.job_name} at step {step.job_name} (index {step_idx})"
                )

        ###### WARNING: No side-effects on self.job object beyond this point. #######

        ## Create local directory for export.
        if not dry:
            self.export_dir.mkdir(parents=True, exist_ok=False)

        ## Materialize sbatch file.
        materialized_sbatch = self.job.materialize()
        if dry:
            logger.warning(f"Dry run. Skipping sbatch materialization at {self.sbatch_file}")
        else:
            self.sbatch_file.write_text(materialized_sbatch)
            self.sbatch_file.chmod(0o755)

        ## Materialize packages to bundle together.
        for package_dir in self.external_package_dirs:
            package_export_dir = self.package_dir / package_dir.name
            if dry:
                logger.warning(f"Dry run. Skipping package sync for {package_dir}")
            else:
                shutil.copytree(
                    package_dir,
                    package_export_dir,
                    ignore=gitignore_filter(
                        package_dir,
                        ignore_files=[
                            package_dir / ".gitignore",
                            package_dir / ".scignore",
                        ],
                        ignore_patterns=[".git/"],
                    ),
                )
                logger.debug(f"Package synced to {package_export_dir}")

        return materialized_sbatch

    def sync(self, host: SlurmSSHRemote | None = None, dry: bool = False) -> dict:
        info = {}

        info["sbatch"] = self.bundle(host=host, dry=dry)
        info["local_dir"] = self.export_dir

        if host:
            info["remote_dir"] = host.sync(self.export_dir, dry=dry)

            if dry:
                logger.warning(f"Skipping sbatch submission to {host.hostname}.")
            else:
                logger.info(f"Submitting sbatch file {host.export_dir / self.export_dir.name / 'sbatch.sh'}")

                slurm_job_id = host.submit(host.export_dir / self.export_dir.name / "sbatch.sh")
                with open(self.export_dir / ".sync" / "sbatch", "w") as f:
                    f.write(str(slurm_job_id))

                logger.info(f"sbatch job {slurm_job_id} submitted")
        else:
            logger.info(f"Job bundle created at {self.export_dir}")

        return info
