"""Shared validation rules for user-authored specifications."""

from io import StringIO

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
