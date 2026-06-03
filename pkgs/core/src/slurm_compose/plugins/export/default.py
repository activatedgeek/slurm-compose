from slurm_compose.api.scripts import PyxisScript, SrunScript
from slurm_compose.config import MOUNT_PATH, logger

from ..base import SlurmComposeExportPlugin


class DefaultExportPlugin(SlurmComposeExportPlugin):
    name = "default"

    def pre_bundle(self, exporter, host, dry: bool = False):
        ## Set job sbatch params and apply overrides from host config when available.
        force_updates = {"job_name": exporter.export_dir.name}
        maybe_updates = {}

        mount_dir = exporter.export_dir

        if host:
            mount_dir = host.export_dir / exporter.export_dir.name

            partition = host.partition

            force_updates |= partition.pop("overrides", {})

            maybe_updates["account"] = host.sbatch_config.get("account")
            maybe_updates |= partition

            if not host.cpu:
                maybe_updates |= {"gpus_per_node": host.gpus_per_node}

        force_updates["output"] = mount_dir / "logs"

        exporter.job.maybe_update(**maybe_updates)
        exporter.job.maybe_update(**force_updates, force=True)

        exporter.job.env["SCOMPOSE_JOB"] = "1"
        exporter.job.env["SCOMPOSE_PKGS"] = f"{mount_dir}/pkgs"
        exporter.job.env["SCOMPOSE_LOGS"] = f"{mount_dir}/logs"

        ## Remove items no longer necessary for steps.
        [force_updates.pop(k, None) for k in ["job_name"]]
        [maybe_updates.pop(k, None) for k in ["account", "partition", "qos", "time"]]

        for step_idx, step in enumerate(exporter.job.steps):
            if isinstance(step, SrunScript):
                step.maybe_update(**maybe_updates)
                step.maybe_update(**force_updates, force=True)

            ## Set bundle mount when available.
            mount_spec = f"{mount_dir}:{MOUNT_PATH}"
            if isinstance(step, PyxisScript) and mount_spec not in step.container_mounts:
                exporter.job.env["SCOMPOSE_PKGS"] = f"{MOUNT_PATH}/pkgs"
                exporter.job.env["SCOMPOSE_LOGS"] = f"{MOUNT_PATH}/logs"
                step.container_mounts += [mount_spec]
                logger.debug(
                    f"Mount {step.container_mounts[-1]} added to {exporter.job.job_name} at step {step.job_name} (index {step_idx})"
                )
