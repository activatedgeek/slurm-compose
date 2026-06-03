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
    def pre_bundle(self, exporter, host: Optional = None, dry: bool = False):
        raise NotImplementedError
