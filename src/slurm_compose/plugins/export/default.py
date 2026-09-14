from typing import ClassVar

from slurm_compose.api.scripts import PyxisScript, SrunScript
from slurm_compose.config import MOUNT_PATH, logger

from ..base import SlurmComposeExportPlugin


class DefaultExportPlugin(SlurmComposeExportPlugin):
    name: ClassVar[str] = "default"

    def pre_bundle(self, host, dry: bool = False):
        if self.exporter.project_name:
            self.exporter.job.env["SCOMPOSE_PROJECT_NAME"] = self.exporter.project_name
        self.exporter.job.env["SCOMPOSE_JOB_NAME"] = self.exporter.job.job_name

        ## Set job sbatch params and apply overrides from host config when available.
        force_updates = {"job_name": self.exporter.export_dir.name}
        maybe_updates = {}

        mount_dir = self.exporter.export_dir

        if host:
            mount_dir = host.export_dir / self.exporter.export_dir.name

            partition = host.partition

            force_updates |= partition.pop("overrides", {})

            maybe_updates["account"] = host.sbatch_config.get("account")
            maybe_updates |= partition

            if not host.cpu:
                maybe_updates |= {"gpus_per_node": host.gpus_per_node}

        force_updates["output"] = mount_dir / "logs"

        self.exporter.job.maybe_update(**maybe_updates)
        self.exporter.job.maybe_update(**force_updates, force=True)

        self.exporter.job.env["SCOMPOSE_PKGS"] = f"{mount_dir}/pkgs"
        self.exporter.job.env["SCOMPOSE_LOGS"] = f"{mount_dir}/logs"

        ## Remove items no longer necessary for steps.
        [force_updates.pop(k, None) for k in ["job_name"]]
        [maybe_updates.pop(k, None) for k in ["account", "partition", "qos", "time"]]

        for step_idx, step in enumerate(self.exporter.job.steps):
            if isinstance(step, SrunScript):
                step.maybe_update(**maybe_updates)
                step.maybe_update(**force_updates, force=True)

            ## Set bundle mount when available.
            mount_spec = f"{mount_dir}:{MOUNT_PATH}"
            if isinstance(step, PyxisScript) and mount_spec not in step.container_mounts:
                self.exporter.job.env["SCOMPOSE_PKGS"] = f"{MOUNT_PATH}/pkgs"
                self.exporter.job.env["SCOMPOSE_LOGS"] = f"{MOUNT_PATH}/logs"
                step.container_mounts += [mount_spec]
                logger.debug(
                    f"Mount {step.container_mounts[-1]} added to {self.exporter.job.job_name} at step {step.job_name} (index {step_idx})"
                )
