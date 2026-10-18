"""A deliberately limited subset of Taskfile schema version 3."""

from graphlib import CycleError, TopologicalSorter
from io import StringIO
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from ruamel.yaml import YAML

NonEmptyString = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Scalar = str | int | float | bool


class TaskfileSpecModel(BaseModel):
    """Reject unsupported fields and implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Task(TaskfileSpecModel):
    """Sequential shell commands, started after all dependencies succeed.

    Commands are expected to terminate. Their runtime behavior is not validated.
    A command list represents separate shell commands, not an argv array.
    """

    desc: str | None = None
    cmds: list[NonEmptyString] = Field(default_factory=list)
    deps: list[NonEmptyString] = Field(default_factory=list)
    vars: dict[str, Scalar] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    dir: NonEmptyString | None = None


class Taskfile(TaskfileSpecModel):
    """Named finite tasks; dependency-only grouping tasks are allowed.

    Variables and template expressions are preserved without evaluation.
    """

    version: Literal["3"]
    tasks: Annotated[dict[NonEmptyString, Task], Field(min_length=1)]
    vars: dict[str, Scalar] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)

    def to_yaml(self) -> str:
        """Serialize the specification as YAML, omitting default-valued fields."""
        output = StringIO()
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.dump(self.model_dump(mode="json", exclude_defaults=True), output)
        return output.getvalue()

    @model_validator(mode="after")
    def validate_dependencies(self) -> Self:
        for name, task in self.tasks.items():
            for dependency in task.deps:
                if dependency not in self.tasks:
                    raise ValueError(f"task {name!r} depends on unknown task {dependency!r}")
        try:
            TopologicalSorter({name: task.deps for name, task in self.tasks.items()}).prepare()
        except CycleError as exc:
            raise ValueError("task dependencies contain a cycle") from exc
        return self
