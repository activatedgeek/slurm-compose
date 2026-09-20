"""Shared validation rules for user-authored specifications."""

import shlex
from io import StringIO
from typing import ClassVar

from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML


class BaseConfig(BaseModel):
    """Reject unknown fields and implicit type coercion in configuration."""

    model_config = ConfigDict(extra="forbid", strict=True)

    def json(self) -> dict:
        """Return a JSON-compatible representation without null fields."""
        return self.model_dump(mode="json", exclude_none=True)

    def yaml(self) -> str:
        """Return the JSON-compatible representation serialized as YAML."""
        output = StringIO()
        YAML(typ="safe").dump(self.json(), output)
        return output.getvalue()


class StepKwargsConfig(BaseConfig):
    """Base configuration for command launcher arguments."""

    _argv_exclude: ClassVar[set[str]] = set()

    @property
    def argv(self) -> list[str]:
        """Return launcher options as equals-separated command arguments."""
        args: list[str] = []
        for name in type(self).model_fields:
            if name in type(self)._argv_exclude:
                continue
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            option = f"--{name.replace('_', '-')}"
            if isinstance(value, bool):
                args.append(option if value else f"--no-{name.replace('_', '-')}")
            elif isinstance(value, list):
                args.append(f"{option}={','.join(str(item) for item in value)}")
            else:
                args.append(f"{option}={value}")
        return args

    def make_script(self, command: str | list[str]) -> str:
        """Convert a shell string or exec-form argv to a renderable command."""
        if isinstance(command, str):
            return f"bash -l {shlex.quote(command)}"
        return shlex.join(command)
