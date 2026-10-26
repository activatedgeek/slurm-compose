import os
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def test_stask_slurm_template():
    template_dir = Path(__file__).parents[1] / "src" / "slurm_compose" / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    script = environment.get_template("slurm.sh.j2").render(
        sbatch_argv=["--job-name=test"],
        taskfile="/shared/run/Taskfile.yml",
    )

    assert "#SBATCH --job-name=test" in script
    assert 'task --silent --exit-code --dir "$taskfile_dir" --taskfile "$taskfile"' in script
    assert 'taskfile="/shared/run/Taskfile.yml"' in script
    assert "maybe_sbatch_requeue" in script
    assert "#SBATCH --signal=B:USR1@10" in script

    result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_task_failure_requeues_within_retry_limit(tmp_path):
    template_dir = Path(__file__).parents[1] / "src" / "slurm_compose" / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    script = environment.get_template("slurm.sh.j2").render(
        sbatch_argv=[],
        taskfile=str(tmp_path / "Taskfile.yml"),
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "scontrol").write_text("""#!/bin/sh
printf '%s\\n' "$*" > "$REQUEUE_LOG"
""")
    (bin_dir / "scontrol").chmod(0o755)
    requeue_log = tmp_path / "requeue.log"
    task_log = tmp_path / "task.log"
    (bin_dir / "task").write_text("""#!/bin/sh
printf '%s\\n' "$*" > "$TASK_LOG"
exit 23
""")
    (bin_dir / "task").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "SLURM_JOB_ID": "1234",
        "SLURM_RESTART_COUNT": "7",
        "STASK_MAX_RESTARTS": "8",
        "REQUEUE_LOG": str(requeue_log),
        "TASK_LOG": str(task_log),
    }

    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)

    assert result.returncode == 23
    assert requeue_log.read_text().strip() == "requeue 1234"
    assert task_log.read_text().strip() == (
        f"--silent --exit-code --dir {tmp_path} --taskfile {tmp_path}/Taskfile.yml"
    )
    assert "attempt 8/8" in result.stderr
