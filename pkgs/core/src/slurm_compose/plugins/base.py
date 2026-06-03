from dataclasses import dataclass
from typing import Any

from slurm_compose.api.scripts import Script


@dataclass(frozen=True)
class SlurmComposePlugin: ...


@dataclass(frozen=True)
class SlurmComposeScriptPlugin(SlurmComposePlugin):
    name: str

    cls: Script

    def __call__(self, *args: Any, **kwargs: Any) -> Script:
        return self.cls(*args, **kwargs)
