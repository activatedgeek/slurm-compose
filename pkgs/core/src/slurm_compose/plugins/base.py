from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from slurm_compose.api.scripts import Script


@dataclass(frozen=True)
class SlurmComposePlugin:
    name: ClassVar[str]


@dataclass(frozen=True)
class SlurmComposeScriptPlugin(SlurmComposePlugin):
    cls: ClassVar[Script]

    def __call__(self, *args: Any, **kwargs: Any) -> Script:
        return self.cls(*args, **kwargs)


@dataclass(frozen=True)
class SlurmComposeExportPlugin(SlurmComposePlugin):
    exporter: Any

    def __post_init__(self):
        if not self.exporter:
            raise ValueError("exporter cannot be undefined.")

    def pre_bundle(self, host: Optional = None, dry: bool = False): ...

    def post_sync(self, host: Optional = None, dry: bool = False): ...
