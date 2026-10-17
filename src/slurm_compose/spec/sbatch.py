import re
import shlex
from collections.abc import Iterator
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from pydantic import Field, PrivateAttr, model_validator

from .base import BaseConfig
from .config import ComposeConfig


class SbatchJobConfig(BaseConfig):
    """Inputs required to render one sbatch job script."""

    compose: ComposeConfig
    extra_argv: list[str] = Field(default_factory=list)
    template_searchpaths: list[str | Path] = Field(default_factory=list)
    _environment: Environment = PrivateAttr()

    @model_validator(mode="after")
    def build_environment(self) -> "SbatchJobConfig":
        self._environment = Environment(
            loader=FileSystemLoader([*self.template_searchpaths, Path(__file__).parent.parent / "templates"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return self

    @property
    def runv(self) -> list[str]:
        """Return the execv-style command used to run this job."""
        return ["sbatch", "--requeue", *self.extra_argv, str(self.job_dir / "sbatch.sh")]

    @property
    def job_name(self) -> str:
        """Return the stable, filesystem-safe job name."""
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.compose.name).strip("-.")
        return f"{name}-{self.compose.hash[:16]}"

    @property
    def job_dir(self) -> Path:
        """Return the stable directory for this compose job configuration."""
        return Path(self.compose.directory) / self.job_name

    @property
    def output(self) -> Path:
        return self.job_dir / "slurm-%j/sbatch.log"

    @property
    def step_topology(self) -> Iterator[dict[str, object]]:
        children = {name: {} for name in self.compose.steps}
        for name, step in self.compose.steps.items():
            for dependency_name in step.depends_on:
                children[dependency_name][name] = {}

        visited: set[str] = set()

        def visit(name: str) -> Iterator[str]:
            if name in visited:
                return
            visited.add(name)
            yield name
            for child in children[name]:
                yield from visit(child)

        roots = (name for name, step in self.compose.steps.items() if not step.depends_on)

        for root in roots:
            for name in visit(root):
                rendered_step = self.compose.steps[name]
                if rendered_step.kwargs is not None:
                    rendered_step = rendered_step.model_copy(
                        update={
                            "kwargs": rendered_step.kwargs.model_copy(
                                update={"output": f'"${{SCOMPOSE_JOB_DIR}}/slurm-%j/{name}.log"'}
                            )
                        }
                    )
                yield {
                    "name": name,
                    "pid_key": shlex.quote(name),
                    "parents": [
                        {"name": shlex.quote(parent), "condition": dependency.condition}
                        for parent, dependency in rendered_step.depends_on.items()
                    ],
                    "environment": {key: shlex.quote(value) for key, value in rendered_step.environment.items()},
                    "scripts": rendered_step.scriptv,
                }

    def script(self, template: str = "slurm.sh.j2") -> str:
        """Render the job as an sbatch script."""
        rendered_sbatch = self.compose.sbatch.model_copy(update={"job_name": self.job_name, "output": self.output})

        template = self._environment.get_template(template)
        return template.render(
            sbatch_argv=rendered_sbatch.argv,
            compose_job_name=shlex.quote(self.compose.name),
            compose_job_dir=shlex.quote(str(self.job_dir)),
            compose_job_hash=self.compose.hash,
            max_restarts=self.compose.sbatch.max_restarts,
            completion_mode=self.compose.sbatch.completion.mode,
            completion_step=(
                None
                if self.compose.sbatch.completion.step is None
                else shlex.quote(self.compose.sbatch.completion.step)
            ),
            steps=self.step_topology,
        )
