"""Shared validation rules for user-authored specifications."""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Reject unknown fields and implicit type coercion in configuration."""

    model_config = ConfigDict(extra="forbid", strict=True)
