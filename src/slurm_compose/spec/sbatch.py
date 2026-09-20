import shlex
from collections.abc import Iterator
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from pydantic import Field, PrivateAttr, computed_field, model_validator

from .base import BaseConfig
from .config import ComposeConfig


class SbatchJobConfig(BaseConfig):
    """Inputs required to render one sbatch job script."""

    compose: ComposeConfig
    passthrough: list[str] = Field(default_factory=list)
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

    @computed_field
    @property
    def runv(self) -> list[str]:
        """Return the execv-style command used to run this job."""
        return ["sbatch", "--requeue", *self.passthrough]

    @property
    def topology(self) -> Iterator[str]:
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
        return (name for root in roots for name in visit(root))

    def script(self, template: str = "slurm.sh.j2") -> str:
        """Render the job as an sbatch script."""
        rendered_steps = []
        for name in self.topology:
            step = self.compose.steps[name]
            rendered_steps.append(
                {
                    "name": name,
                    "pid_key": shlex.quote(name),
                    "parents": [shlex.quote(parent) for parent in step.depends_on],
                    "environment": {key: shlex.quote(value) for key, value in step.environment.items()},
                    "scripts": step.scriptv,
                }
            )

        template = self._environment.get_template(template)
        return template.render(
            sbatch_argv=self.compose.sbatch.argv,
            config_hash=self.compose.hash,
            steps=rendered_steps,
        )
